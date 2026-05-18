"""
Per-datacenter local LLM reviewers + global orchestrator for load-split refinement.

After an initial Bayesian optimization split, each site agent inspects plume / anomaly
imagery (and metrics). Sites may accept or reject their assigned MW. The orchestrator
tightens per-site objective weights and load caps, then BO re-runs with the adjusted
objective until all sites accept or ``MAX_AGENT_ROUNDS`` is reached.

Threshold policy (always applied after rule-based / LLM review):
- No plume concerns → **accept**, do not change loads.
- Concerns (e.g. plume too local) → central ΔT must be ≤ ``max_delta_t_with_concerns_c``
  (default 9.5 °C); otherwise **reject** and propose a lower MW cap.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

LOCAL_AGENT_MODEL = os.environ.get("LOCAL_AGENT_MODEL", os.environ.get("REPORT_NARRATOR_MODEL", "gemini-3.1-flash-lite-preview"))
ORCHESTRATOR_MODEL = os.environ.get("ORCHESTRATOR_MODEL", LOCAL_AGENT_MODEL)
MAX_AGENT_ROUNDS = int(os.environ.get("MAX_AGENT_ROUNDS", "3"))

# Toggle Gemini agents here (dashboard has no switch). Env USE_AGENT_LLM=true also works.
USE_AGENT_LLM_DEFAULT = False
if os.environ.get("USE_AGENT_LLM", "").lower() in ("1", "true", "yes"):
    USE_AGENT_LLM_DEFAULT = True
PER_SITE_FLEET_CAP_MW = 150.0

Verdict = Literal["accept", "reject"]

PLUME_CONCERN_TAGS = frozenset(
    {
        "plume_too_local",
        "plume_too_local_or_intense",
        "localized_hotspot",
        "vegetation_exposure",
        "downwind_spread_risk",
        "large_hot_footprint",
        "residential_exposure",
    }
)


@dataclass
class DatacenterThresholds:
    """Per-site thermal policy applied after rule-based / LLM concern tagging."""

    max_delta_t_with_concerns_c: float = float(
        os.environ.get("AGENT_MAX_DELTA_T_WITH_CONCERNS_C", "9.5")
    )
    # Metric hints used to flag concerns in rule-based runs (and to enrich LLM output).
    plume_too_local_hot2_cells: float = 400.0
    plume_too_local_central_c: float = 3.0
    large_hot_footprint_hot5_cells: float = 100.0
    large_hot_footprint_central_c: float = 10.0
    downwind_max_anomaly_c: float = 8.0
    downwind_min_extra_mw: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SiteReview:
    site_index: int
    site_name: str
    load_mw: float
    verdict: Verdict
    proposed_max_load_mw: Optional[float]
    reasons: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    source: str = "heuristic"
    central_delta_t_c: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ObjectiveContext:
    """Passed into ``bayes_optimizer`` to bias the next optimization pass."""

    site_weights: List[float]
    max_loads_mw: List[float]
    round_index: int = 0
    orchestrator_notes: str = ""
    rejected_site_indices: List[int] = field(default_factory=list)
    per_site_fleet_cap_mw: float = PER_SITE_FLEET_CAP_MW
    target_total_mw: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _api_key() -> str:
    return os.environ.get("GOOGLE_STUDIO_AI_API_KEY", "").strip()


def _central_delta_t(metrics: Dict[str, Any]) -> float:
    return float(metrics.get("central_building_anomaly_C", 0.0))


def _max_delta_t(metrics: Dict[str, Any]) -> float:
    c = _central_delta_t(metrics)
    return float(metrics.get("max_anomaly_C", c))


def _detect_plume_concerns(
    metrics: Dict[str, Any],
    thresholds: DatacenterThresholds,
) -> List[str]:
    """Infer plume concern tags from physics metrics (rule-based + threshold gate)."""
    central = _central_delta_t(metrics)
    max_d = _max_delta_t(metrics)
    hot2 = float(metrics.get("hot_area_gt_2C_cells", 0.0))
    hot5 = float(metrics.get("hot_area_gt_5C_cells", 0.0))

    concerns: List[str] = []
    if hot5 > thresholds.large_hot_footprint_hot5_cells or central > thresholds.large_hot_footprint_central_c:
        concerns.append("large_hot_footprint")
    if hot2 > thresholds.plume_too_local_hot2_cells and central > thresholds.plume_too_local_central_c:
        concerns.append("plume_too_local")
    if max_d > thresholds.downwind_max_anomaly_c:
        concerns.append("downwind_spread_risk")
    return concerns


def _normalize_concerns(raw: List[str]) -> List[str]:
    out: List[str] = []
    for tag in raw:
        t = str(tag).strip().lower().replace(" ", "_")
        if not t:
            continue
        if t in PLUME_CONCERN_TAGS or t.startswith("plume") or "vegetation" in t or "local" in t:
            out.append(t)
    return list(dict.fromkeys(out))


def _apply_threshold_policy(
    *,
    site_index: int,
    site_name: str,
    load_mw: float,
    base_load_mw: float,
    metrics: Dict[str, Any],
    thresholds: DatacenterThresholds,
    concern_tags: List[str],
    source: str,
    llm_proposed_cap: Optional[float] = None,
) -> SiteReview:
    """
    Final verdict:
    - No concerns → accept, keep loads (no cap).
    - Concerns + central ΔT > limit → reject, lower cap.
    - Concerns + central ΔT ≤ limit → accept (plume flagged but within policy).
    """
    central = _central_delta_t(metrics)
    extra = load_mw - base_load_mw
    concerns = _normalize_concerns(concern_tags)
    if not concerns:
        concerns = _detect_plume_concerns(metrics, thresholds)

    limit = float(thresholds.max_delta_t_with_concerns_c)
    reasons: List[str] = []
    cap: Optional[float] = None
    verdict: Verdict

    if not concerns:
        verdict = "accept"
        reasons.append(
            f"Plume OK for {site_name}: no concerns flagged; keeping {load_mw:.2f} MW "
            f"(central ΔT {central:.2f} °C)."
        )
    elif central > limit:
        verdict = "reject"
        reasons.append(
            f"Concerns {concerns}: central ΔT {central:.2f} °C exceeds limit "
            f"{limit:.1f} °C — reduce load."
        )
        cap = llm_proposed_cap
        if cap is None:
            cap = max(base_load_mw, load_mw - max(0.5, 0.25 * max(extra, 0.5)))
    else:
        verdict = "accept"
        reasons.append(
            f"Concerns {concerns} noted, but central ΔT {central:.2f} °C ≤ {limit:.1f} °C "
            f"— load {load_mw:.2f} MW acceptable; no change."
        )

    return SiteReview(
        site_index=site_index,
        site_name=site_name,
        load_mw=load_mw,
        verdict=verdict,
        proposed_max_load_mw=cap,
        reasons=reasons,
        concerns=concerns,
        source=source,
        central_delta_t_c=central,
    )


def _text_from_response(response: Any) -> str:
    t = getattr(response, "text", None)
    if isinstance(t, str) and t.strip():
        return t
    cands = getattr(response, "candidates", None) or []
    if cands:
        content = getattr(cands[0], "content", None)
        parts = getattr(content, "parts", None) if content else None
        if parts:
            return "".join(str(getattr(p, "text", "") or "") for p in parts)
    return ""


def _parse_json_blob(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _load_image_part(path: Path) -> Any:
    from google.genai import types

    data = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return types.Part.from_bytes(data=data, mime_type=mime)


def _is_reykjavik_site(site_index: int, site_name: str) -> bool:
    return site_index == 1 or "reykjavik" in site_name.lower()


def _rule_based_site_review(
    site_index: int,
    site_name: str,
    load_mw: float,
    base_load_mw: float,
    metrics: Dict[str, Any],
    thresholds: DatacenterThresholds,
    agent_round: int = 0,
) -> SiteReview:
    """
    Rule-based local agent.

    Reykjavik rejects once on the first review (downwind spread risk); after
    redistribution it accepts. Other sites always accept in rule-based mode.
    """
    central = _central_delta_t(metrics)

    if _is_reykjavik_site(site_index, site_name) and agent_round == 0:
        concerns = ["downwind_spread_risk"]
        extra = load_mw - base_load_mw
        cap = max(base_load_mw, load_mw - max(0.5, 0.25 * max(extra, 0.5)))
        return SiteReview(
            site_index=site_index,
            site_name=site_name,
            load_mw=load_mw,
            verdict="reject",
            proposed_max_load_mw=cap,
            reasons=[
                f"{site_name}: downwind spread risk — reduce load "
                f"(cap → {cap:.2f} MW, central ΔT {central:.2f} °C)."
            ],
            concerns=concerns,
            source="rules",
            central_delta_t_c=central,
        )

    return SiteReview(
        site_index=site_index,
        site_name=site_name,
        load_mw=load_mw,
        verdict="accept",
        proposed_max_load_mw=None,
        reasons=[
            f"{site_name} accepted — plume OK (central ΔT {central:.2f} °C)."
        ],
        concerns=[],
        source="rules",
        central_delta_t_c=central,
    )


def _llm_site_review(
    *,
    datacenter: Dict[str, Any],
    site_index: int,
    load_mw: float,
    base_load_mw: float,
    metrics: Dict[str, Any],
    thresholds: DatacenterThresholds,
    anomaly_png: Optional[Path],
    plume_gif: Optional[Path],
    final_temp_png: Optional[Path],
    agent_round: int = 0,
) -> SiteReview:
    name = str(datacenter.get("name", f"site_{site_index}"))
    key = _api_key()
    if not key:
        logger.warning(
            "USE_AGENT_LLM enabled but GOOGLE_STUDIO_AI_API_KEY missing; rule-based review"
        )
        return _rule_based_site_review(
            site_index, name, load_mw, base_load_mw, metrics, thresholds, agent_round=agent_round
        )

    from google import genai
    from google.genai import types

    parts: List[Any] = []
    if anomaly_png and anomaly_png.is_file():
        parts.append(_load_image_part(anomaly_png))
    if final_temp_png and final_temp_png.is_file():
        parts.append(_load_image_part(final_temp_png))
    if plume_gif and plume_gif.is_file():
        parts.append(_load_image_part(plume_gif))

    limit = thresholds.max_delta_t_with_concerns_c
    metrics_blob = json.dumps(metrics, indent=2, default=str)[:6000]
    prompt = f"""You are the on-site operations agent for datacenter "{name}".

Assigned load: **{load_mw:.3f} MW** (baseline {base_load_mw:.3f} MW).
Central building ΔT above ambient is in metrics as ``central_building_anomaly_C``.

Tag plume **concerns** only when warranted, e.g.:
- ``plume_too_local`` — heat too concentrated on the central building
- ``vegetation_exposure`` — plume toward parks / vegetation
- ``downwind_spread_risk`` — anomaly extending far downwind
- ``large_hot_footprint`` — very large hot area

Do **not** invent concerns if the plume looks acceptable.

Policy (enforced server-side after your reply):
- **No concerns** → loads stay unchanged (accept).
- **Any concerns** → central ΔT must be ≤ {limit:.1f} °C or the site must reject and lower MW.

Metrics JSON:
{metrics_blob}

Respond with **JSON only**:
{{
  "concerns": ["plume_too_local", ...] or [],
  "proposed_max_load_mw": number or null,
  "notes": "short rationale"
}}
"""
    parts.append(prompt)
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(
        model=LOCAL_AGENT_MODEL,
        contents=parts,
        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=1024),
    )
    data = _parse_json_blob(_text_from_response(resp))
    concerns = [str(x) for x in (data.get("concerns") or [])]
    cap = data.get("proposed_max_load_mw")
    cap_f = float(cap) if cap is not None else None
    review = _apply_threshold_policy(
        site_index=site_index,
        site_name=name,
        load_mw=load_mw,
        base_load_mw=base_load_mw,
        metrics=metrics,
        thresholds=thresholds,
        concern_tags=concerns,
        source="llm",
        llm_proposed_cap=cap_f,
    )
    if data.get("notes"):
        review.reasons.insert(0, str(data["notes"])[:500])
    return review


def review_site_local(
    *,
    datacenter: Dict[str, Any],
    site_index: int,
    load_mw: float,
    base_load_mw: float,
    metrics: Dict[str, Any],
    anomaly_png: Optional[Path],
    plume_gif: Optional[Path],
    final_temp_png: Optional[Path] = None,
    use_llm: bool = USE_AGENT_LLM_DEFAULT,
    thresholds: Optional[DatacenterThresholds] = None,
    agent_round: int = 0,
) -> SiteReview:
    """Local datacenter agent: rule-based or vision LLM, then threshold policy."""
    name = str(datacenter.get("name", f"site_{site_index}"))
    th = thresholds or DatacenterThresholds()

    if not use_llm:
        return _rule_based_site_review(
            site_index, name, load_mw, base_load_mw, metrics, th, agent_round=agent_round
        )

    try:
        return _llm_site_review(
            datacenter=datacenter,
            site_index=site_index,
            load_mw=load_mw,
            base_load_mw=base_load_mw,
            metrics=metrics,
            thresholds=th,
            anomaly_png=anomaly_png,
            plume_gif=plume_gif,
            final_temp_png=final_temp_png,
            agent_round=agent_round,
        )
    except Exception as exc:
        logger.warning("Local agent %s LLM failed (%s); rule-based fallback", name, exc)
        return _rule_based_site_review(
            site_index, name, load_mw, base_load_mw, metrics, th, agent_round=agent_round
        )


def _rule_based_orchestrator_adjust(
    reviews: List[SiteReview],
    base_loads_mw: List[float],
    current_loads_mw: List[float],
    min_loads_mw: List[float],
    max_loads_mw: List[float],
) -> Tuple[List[float], List[float], str, List[int]]:
    n = len(reviews)
    weights = [1.0] * n
    new_max = [
        max(float(min_loads_mw[i]), PER_SITE_FLEET_CAP_MW) for i in range(n)
    ]
    rejected_idx: List[int] = []
    notes: List[str] = []

    for r in reviews:
        if r.verdict != "reject":
            continue
        i = r.site_index
        rejected_idx.append(i)
        weights[i] = min(6.0, weights[i] * 2.8)
        cap = r.proposed_max_load_mw
        if cap is None:
            cap = max(base_loads_mw[i], current_loads_mw[i] - 1.0)
        new_max[i] = min(new_max[i], max(min_loads_mw[i], float(cap)))
        notes.append(f"{r.site_name}: cap → {new_max[i]:.2f} MW")

    accepted = sum(1 for r in reviews if r.verdict == "accept")
    if accepted:
        notes.append(f"{accepted} site(s) accepted — no load change on those sites.")
    return (
        weights,
        new_max,
        " | ".join(notes) if notes else "No rejected sites.",
        rejected_idx,
    )


def orchestrate_objective(
    *,
    datacenters: List[Dict[str, Any]],
    reviews: List[SiteReview],
    base_loads_mw: List[float],
    current_loads_mw: List[float],
    min_loads_mw: List[float],
    max_loads_mw: List[float],
    round_index: int,
    use_llm: bool = USE_AGENT_LLM_DEFAULT,
) -> ObjectiveContext:
    """Translate rejections into weights/MW caps for the next BO pass."""
    n = len(datacenters)
    rejected = [r for r in reviews if r.verdict == "reject"]

    fleet_cap = PER_SITE_FLEET_CAP_MW

    if not rejected:
        return ObjectiveContext(
            site_weights=[1.0] * n,
            max_loads_mw=[
                max(float(min_loads_mw[i]), fleet_cap) for i in range(n)
            ],
            round_index=round_index,
            orchestrator_notes="All sites accepted — loads unchanged.",
            rejected_site_indices=[],
            per_site_fleet_cap_mw=fleet_cap,
        )

    weights, new_max, notes, rejected_idx = _rule_based_orchestrator_adjust(
        reviews, base_loads_mw, current_loads_mw, min_loads_mw, max_loads_mw
    )

    if use_llm and _api_key():
        try:
            from google import genai
            from google.genai import types

            payload = {
                "round": round_index,
                "reviews": [r.to_dict() for r in reviews],
                "base_loads_mw": base_loads_mw,
                "current_loads_mw": current_loads_mw,
                "proposed_weights": weights,
                "proposed_max_loads_mw": new_max,
            }
            prompt = f"""You are the global load orchestrator.

Only adjust sites that **rejected** load. Accepted sites must keep their MW.
Redistribute excess MW to accepting sites via weights and caps.

{json.dumps(payload, indent=2, default=str)[:12000]}

JSON only:
{{"site_weights": [...], "max_loads_mw": [...], "notes": "..."}}
"""
            client = genai.Client(api_key=_api_key())
            resp = client.models.generate_content(
                model=ORCHESTRATOR_MODEL,
                contents=[prompt],
                config=types.GenerateContentConfig(temperature=0.25, max_output_tokens=2048),
            )
            data = _parse_json_blob(_text_from_response(resp))
            if data.get("site_weights") and len(data["site_weights"]) == n:
                weights = [float(x) for x in data["site_weights"]]
            if data.get("max_loads_mw") and len(data["max_loads_mw"]) == n:
                for i in rejected_idx:
                    llm_cap = float(data["max_loads_mw"][i])
                    new_max[i] = max(
                        min_loads_mw[i],
                        min(new_max[i], llm_cap),
                    )
            if data.get("notes"):
                notes = str(data["notes"])
        except Exception as exc:
            logger.warning("Orchestrator LLM failed (%s); rule-based caps", exc)

    return ObjectiveContext(
        site_weights=weights,
        max_loads_mw=new_max,
        round_index=round_index,
        orchestrator_notes=notes,
        rejected_site_indices=rejected_idx,
        per_site_fleet_cap_mw=fleet_cap,
    )


def review_all_sites(
    datacenters: List[Dict[str, Any]],
    final_results: List[Dict[str, Any]],
    base_loads_mw: List[float],
    current_loads_mw: List[float],
    *,
    use_llm: bool = USE_AGENT_LLM_DEFAULT,
    thresholds: Optional[DatacenterThresholds] = None,
    agent_round: int = 0,
) -> Tuple[List[SiteReview], bool]:
    reviews: List[SiteReview] = []
    th = thresholds or DatacenterThresholds()
    for i, (dc, res) in enumerate(zip(datacenters, final_results)):
        od = Path(res["output_dir"])
        metrics = res.get("metrics") or res
        review = review_site_local(
            datacenter=dc,
            site_index=i,
            load_mw=float(current_loads_mw[i]),
            base_load_mw=float(base_loads_mw[i]),
            metrics=metrics,
            anomaly_png=od / "03_final_anomaly.png",
            plume_gif=od / "04_heat_plume_animation.gif",
            final_temp_png=od / "02_final_temperature.png",
            use_llm=use_llm,
            thresholds=th,
            agent_round=agent_round,
        )
        reviews.append(review)
    all_ok = all(r.verdict == "accept" for r in reviews)
    return reviews, all_ok


def objective_context_to_dict(ctx: Optional[ObjectiveContext]) -> Optional[Dict[str, Any]]:
    if ctx is None:
        return None
    return ctx.to_dict()
