"""
LLM layer for optimization report transparency.

Builds a plain-text explanation from ``optimal_data.json``-shaped payloads (Bayesian
optimization traces, datacenter parameters) and converts it to safe HTML for
embedding in ``report.html``.

Requires ``GOOGLE_STUDIO_AI_API_KEY`` for the live model; otherwise emits a
deterministic fallback summary from the numbers alone.
"""

from __future__ import annotations

import json
import logging
import os
import re
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

NARRATOR_MODEL = os.environ.get("REPORT_NARRATOR_MODEL", "gemini-3.1-flash-lite-preview")

OBJECTIVE_HINT = """Global objective (lower is better) is the scalar optimized by the run. It penalizes:
(1) a smooth maximum of per-site max thermal anomaly vs ambient (so one overloaded site hurts a lot),
(2) standard deviation of max anomalies and of central-building anomalies across sites (spread of stress),
(3) sums of max and mean anomalies,
(4) hot-area cell counts (footprint of >1°C, >2°C, >5°C anomalies, weighted),
(5) a mild smooth worst of absolute peak temperature (°C).
The JSON field site_objective per site is a diagnostic only (2*max_delta + central + 0.5*mean + hot penalty); it is not additive into the global scalar."""


def _sanitize_datacenters(datacenters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = (
        "name",
        "lat",
        "lon",
        "temp_c",
        "humidity",
        "solar_wm2",
        "wind_speed_m_s",
        "wind_direction",
        "wall_material",
        "wall_specific_heat_kj_per_kg_k",
        "wall_density_kg_m3",
    )
    out: List[Dict[str, Any]] = []
    for dc in datacenters:
        row = {k: dc.get(k) for k in keys if k in dc}
        out.append(row)
    return out


def _text_from_genai_response(response: Any) -> str:
    t = getattr(response, "text", None)
    if isinstance(t, str) and t.strip():
        return t
    gens = getattr(response, "generations", None)
    if gens and getattr(gens[0], "text", None):
        return str(gens[0].text)
    cands = getattr(response, "candidates", None) or []
    if cands:
        content = getattr(cands[0], "content", None)
        parts = getattr(content, "parts", None) if content else None
        if parts:
            return "".join(str(getattr(p, "text", "") or "") for p in parts)
    return ""


def _plain_to_safe_html(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "<p><em>(Empty narrative.)</em></p>"
    blocks: List[str] = []
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if not para:
            continue
        if para.startswith("## "):
            blocks.append(f"<h3>{escape(para[3:].strip())}</h3>")
            continue
        lines = [ln for ln in para.split("\n") if ln.strip()]
        if lines and all(
            ln.strip().startswith(("- ", "* ", "• "))
            for ln in lines
        ):
            lis = []
            for ln in lines:
                s = ln.strip()
                s = re.sub(r"^[\-\*•]\s*", "", s)
                lis.append(f"<li>{escape(s)}</li>")
            blocks.append("<ul>" + "".join(lis) + "</ul>")
        else:
            inner = escape(para).replace("\n", "<br />\n")
            blocks.append(f"<p>{inner}</p>")
    return "\n".join(blocks) if blocks else "<p><em>(Empty narrative.)</em></p>"


def _fallback_narrative(bundle: Dict[str, Any], error: Optional[str] = None) -> str:
    gs = list(bundle.get("global_search") or [])
    rf = bundle.get("refinement") or []
    best_o = bundle.get("best_objective")
    base = bundle.get("base_loads_mw") or []
    best = bundle.get("best_loads_mw") or []
    bps = bundle.get("best_per_site") or []
    parts: List[str] = []

    parts.append("## What was tested")
    parts.append(
        f"- Recorded {len(gs)} global physics evaluations (each row: one feasible MW split + resulting objective)."
    )
    if best_o is not None:
        parts.append(f"- Best objective after the final full-resolution rerun: {best_o} (lower is better).")

    if gs:
        by_o = sorted(gs, key=lambda r: float(r["objective"]))
        worst = by_o[-1]
        parts.append(
            f"- Best candidate in trace: index {by_o[0].get('candidate_index')} objective {float(by_o[0]['objective']):.4f} loads MW {by_o[0].get('loads_mw')}."
        )
        parts.append(
            f"- Worst in trace: index {worst.get('candidate_index')} objective {float(worst['objective']):.4f} loads MW {worst.get('loads_mw')}."
        )

    parts.append("## What the metric means here")
    parts.append(OBJECTIVE_HINT.replace("\n", " "))
    parts.append(
        "- A lower objective usually means less concentrated overheating, less spread between the hottest sites, and smaller hot plumes—not merely lower average load."
    )

    eff_max = bundle.get("effective_max_loads_mw")
    if eff_max and base and len(eff_max) == len(base):
        parts.append("## Agent MW caps (orchestrator)")
        for i, cap in enumerate(eff_max):
            cap_f = float(cap)
            if cap_f < 149.0:
                parts.append(
                    f"- Site {i + 1}: capped at {cap_f:.3f} MW after local agent rejection."
                )

    parts.append("## Chosen loads vs baseline")
    if base and best and len(base) == len(best):
        parts.extend(
            [
                f"- Site {i + 1}: {float(b):.3f} → {float(x):.3f} MW (Δ {float(x) - float(b):+.3f})"
                for i, (b, x) in enumerate(zip(base, best))
            ]
        )

    parts.append("## Per-site outcome at optimum (from best_per_site)")
    for i, row in enumerate(bps):
        m = row.get("metrics") or {}
        so = row.get("site_objective")
        parts.append(
            f"- Site {i + 1}: diagnostic site_objective={so}; "
            f"central ΔT anomaly {float(m.get('central_building_anomaly_C', 0)):.3f} °C; "
            f"max anomaly {float(m.get('max_anomaly_C', m.get('central_building_anomaly_C', 0))):.3f} °C."
        )

    if rf:
        parts.append("## Local refinement (coordinate polish on top seeds)")
        for r in rf:
            b0 = float(r.get("initial_objective", 0))
            af = float(r.get("objective", 0))
            parts.append(
                f"- Seed rank {int(r.get('initial_rank', 0)) + 1}: objective {b0:.4f} → {af:.4f} (Δ {af - b0:+.4f})."
            )

    parts.append("## Uncertainty (read the GP chart if present)")
    parts.append(
        "- Wider ±σ bands on the surrogate plot mean the cheap GP model disagrees with itself at those evaluated latent codes; they are not a calibrated physics noise interval."
    )

    if error:
        parts.append("## Note")
        parts.append(f"- Narrative source: offline fallback ({error}).")

    return "\n\n".join(parts)


def generate_optimization_narrative_html(
    datacenters: List[Dict[str, Any]],
    optimal_payload: Dict[str, Any],
    *,
    run_root: Optional[Path] = None,
) -> str:
    """
    Produce HTML (safe fragment) explaining the optimization run.

    Saves raw model text to ``narrative_llm.txt`` under ``run_root`` when provided.
    """
    key = os.environ.get("GOOGLE_STUDIO_AI_API_KEY", "").strip()
    global_rows = list(optimal_payload.get("global_search") or [])
    bundle_for_prompt: Dict[str, Any] = {
        "run_id": optimal_payload.get("run_id"),
        "extra_total_load_mw": optimal_payload.get("extra_total_load_mw"),
        "base_loads_mw": optimal_payload.get("base_loads_mw"),
        "best_loads_mw": optimal_payload.get("best_loads_mw"),
        "best_objective": optimal_payload.get("best_objective"),
        "optimization_method": optimal_payload.get("optimization_method"),
        "bo_init_evals": optimal_payload.get("bo_init_evals"),
        "bo_ei_evals": optimal_payload.get("bo_ei_evals"),
        "bo_latent_dim": optimal_payload.get("bo_latent_dim"),
        "target_total_mw": optimal_payload.get("target_total_mw"),
        "cache_size": optimal_payload.get("cache_size"),
        "random_seed": optimal_payload.get("random_seed"),
        "charts": optimal_payload.get("charts"),
        "global_search": global_rows[:50],
        "global_search_truncated": len(global_rows) > 50,
        "refinement": optimal_payload.get("refinement") or [],
        "best_per_site": optimal_payload.get("best_per_site") or [],
        "datacenters": _sanitize_datacenters(datacenters),
    }
    json_blob = json.dumps(bundle_for_prompt, indent=2, default=str)
    if len(json_blob) > 100_000:
        json_blob = json_blob[:100_000] + "\n… [truncated for prompt size]"

    prompt = f"""You are writing an operations-facing report for colleagues who already ran this optimization.

Output **plain text only**: double newlines between paragraphs, section titles as lines starting with "## ", bullets as lines starting with "- ". No HTML, no markdown code fences.

Do **not** explain Bayesian optimization, Gaussian processes, softmax latent codes, or expected improvement as a tutorial. At most **one short sentence** may name the optimizer if needed to interpret a chart filename.

You **must** use the JSON numbers and site names. Cover:

1) **What was tested** — How many distinct MW splits were simulated? Name the best and worst few by objective (candidate_index, objective, loads_mw). What total MW was being redistributed (from base_loads_mw, best_loads_mw, extra_total_load_mw / target_total_mw if present)?

2) **Why the winning split is lower objective** — Tie best_objective to best_per_site metrics (central_building_anomaly_C, max_anomaly_C / max_temp_C, hot-area fields). For at least one high-objective candidate in global_search, say which site or metric likely drove the worse score.

3) **Load allocation narrative** — Why does best_loads_mw put more MW on some sites and less on others relative to poor candidates? Reference datacenter fields (lat/lon, temp, wind) only when they plausibly support the thermal outcome.

4) **Where uncertainty is larger** — From the optimization trace and (if you infer from text about charts) the surrogate σ band: when does the model disagree with itself or when were evaluations far from the running best? Clarify that GP ±σ is **surrogate uncertainty on latent load codes**, not Monte Carlo noise from the physics simulator.

5) **Metric definition** — One concise paragraph: global objective is a single scalar (lower better) combining worst-site thermal stress, cross-site imbalance, summed anomalies, hot plume footprint, and mild worst absolute temperature pressure. Explain that per-site `site_objective` in JSON is a **diagnostic**, not a sum that rebuilds the global score.

6) **Refinement** — For each refinement row, state initial vs final objective and whether polish materially helped.

Reference chart keys from "charts" (e.g. improvement_trace, gp_surrogate, load_heatmap, site_scores, refinement) only to point readers to visuals, not to re-teach ML.

OBJECTIVE REFERENCE (do not contradict):
{OBJECTIVE_HINT}

DATA (JSON):
{json_blob}
"""

    raw_text = ""
    if not key:
        raw_text = _fallback_narrative(bundle_for_prompt, error="GOOGLE_STUDIO_AI_API_KEY not set")
        logger.info("Optimization narrative: using fallback (no API key)")
    else:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=key)
            logger.info("Optimization narrative: calling model=%s", NARRATOR_MODEL)
            resp = client.models.generate_content(
                model=NARRATOR_MODEL,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0.35,
                    max_output_tokens=8192,
                ),
            )
            raw_text = _text_from_genai_response(resp).strip()
            if not raw_text:
                raw_text = _fallback_narrative(
                    bundle_for_prompt, error="Empty model response"
                )
        except Exception as exc:
            logger.exception("Optimization narrative LLM failed: %s", exc)
            raw_text = _fallback_narrative(bundle_for_prompt, error=str(exc))

    if run_root is not None:
        try:
            run_root.mkdir(parents=True, exist_ok=True)
            p = run_root / "narrative_llm.txt"
            p.write_text(raw_text, encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not write narrative_llm.txt: %s", exc)

    inner = _plain_to_safe_html(raw_text)
    return f'<div class="narrative-body">{inner}</div>'
