from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import json
import numpy as np

from run_one_physical_simulation import (
    DEFAULT_SAVE_EVERY,
    DEFAULT_STEPS,
    run_physical_simulation_for_params,
)


# ============================================================
# 1. CONFIG
# ============================================================

# Defaults for standalone CLI demos (MW).
BASE_LOAD_MW = 2.0
EXTRA_TOTAL_LOAD_MW = 5.0

MIN_LOAD_MW = 1.0
MAX_LOAD_MW = 150.0

# Fewer global random splits and shallower refinement (faster).
N_GLOBAL_CANDIDATES = 12
TOP_K_REFINE = 2

RANDOM_SEED = 42

# Fast physics during search (full run uses DEFAULT_STEPS from simulation).
FAST_PHYSICS_STEPS = 700
FAST_PHYSICS_SAVE_EVERY = 120

# For optimization, keep outputs light.
# After best load is found, you can rerun best case with save_gif=True.
OPTIMIZATION_SAVE_GIF = False

OUTPUT_ROOT = Path("load_optimization_outputs")


# ============================================================
# 2. OBJECTIVE
# ============================================================

def compute_single_datacenter_objective(metrics: Dict[str, Any]) -> float:
    """
    Lower is better.

    This objective prioritizes:
        - absolute max temperature
        - central building temperature
        - hot area above absolute thresholds
        - existing thermal_risk_objective if available

    This is better than only using temperature anomaly because for datacenter
    operation, absolute temperature matters.
    """
    max_temp = float(metrics.get("max_temp_C", 0.0))
    central_temp = float(metrics.get("central_building_temperature_C", max_temp))
    mean_temp = float(metrics.get("mean_temp_C", max_temp))

    hot30 = int(metrics.get("hot_area_gt_30C_cells", 0))
    hot35 = int(metrics.get("hot_area_gt_35C_cells", 0))
    hot40 = int(metrics.get("hot_area_gt_40C_cells", 0))
    hot45 = int(metrics.get("hot_area_gt_45C_cells", 0))

    # Use built-in thermal risk if available, but add central temp explicitly.
    base_risk = float(metrics.get("thermal_risk_objective", max_temp))

    objective = (
        1.0 * base_risk
        + 0.35 * central_temp
        + 0.05 * mean_temp
        + 0.002 * hot30
        + 0.010 * hot35
        + 0.030 * hot40
        + 0.100 * hot45
    )

    return float(objective)


def compute_total_objective(results: List[Dict[str, Any]]) -> float:
    """
    Total load-allocation objective.

    We minimize the sum of all site risks.
    """
    total = 0.0

    for result in results:
        metrics = result["metrics"]
        total += compute_single_datacenter_objective(metrics)

    return float(total)


# ============================================================
# 3. LOAD PROJECTION HELPERS
# ============================================================

def project_loads_to_bounds(
    loads: np.ndarray,
    min_loads: np.ndarray,
    max_loads: np.ndarray,
    target_total_load: float,
    max_iter: int = 100,
) -> np.ndarray:
    """
    Project loads into bounds while preserving total load approximately.
    """
    loads = np.clip(loads.astype(float).copy(), min_loads, max_loads)

    for _ in range(max_iter):
        diff = target_total_load - loads.sum()

        if abs(diff) < 1e-10:
            break

        if diff > 0:
            free = loads < max_loads - 1e-12

            if not np.any(free):
                break

            capacity = np.where(free, max_loads - loads, 0.0)
            capacity_sum = capacity.sum()

            if capacity_sum <= 1e-12:
                break

            add = diff * capacity / capacity_sum
            loads += np.minimum(add, capacity)

        else:
            free = loads > min_loads + 1e-12

            if not np.any(free):
                break

            removable = np.where(free, loads - min_loads, 0.0)
            removable_sum = removable.sum()

            if removable_sum <= 1e-12:
                break

            remove_amount = -diff
            remove = remove_amount * removable / removable_sum
            loads -= np.minimum(remove, removable)

    return np.clip(loads, min_loads, max_loads)


def generate_candidate_loads(
    n_datacenters: int,
    base_loads: np.ndarray,
    min_loads: np.ndarray,
    max_loads: np.ndarray,
    extra_total_load_mw: float,
    n_candidates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate candidate load distributions.

    Includes:
        - equal split
        - all extra load to one site
        - random Dirichlet splits
        - cooling-score-biased splits can be added outside if desired
    """
    target_total_load = float(base_loads.sum() + extra_total_load_mw)

    candidates = []

    # Equal split.
    equal_extra = np.ones(n_datacenters) * (extra_total_load_mw / n_datacenters)
    candidates.append(
        project_loads_to_bounds(
            loads=base_loads + equal_extra,
            min_loads=min_loads,
            max_loads=max_loads,
            target_total_load=target_total_load,
        )
    )

    # All extra load to one datacenter.
    for i in range(n_datacenters):
        extra = np.zeros(n_datacenters)
        extra[i] = extra_total_load_mw

        candidates.append(
            project_loads_to_bounds(
                loads=base_loads + extra,
                min_loads=min_loads,
                max_loads=max_loads,
                target_total_load=target_total_load,
            )
        )

    # Random load splits.
    for _ in range(n_candidates):
        if rng.random() < 0.5:
            alpha = np.ones(n_datacenters) * 0.7
        else:
            alpha = np.ones(n_datacenters) * 2.0

        extra = rng.dirichlet(alpha) * extra_total_load_mw

        candidates.append(
            project_loads_to_bounds(
                loads=base_loads + extra,
                min_loads=min_loads,
                max_loads=max_loads,
                target_total_load=target_total_load,
            )
        )

    # Remove duplicates.
    unique = []
    seen = set()

    for candidate in candidates:
        key = tuple(np.round(candidate, 4))

        if key not in seen:
            seen.add(key)
            unique.append(candidate)

    return np.array(unique)


# ============================================================
# 4. CACHE
# ============================================================

class SimulationCache:
    """
    Cache simulations by datacenter index and rounded load.

    This avoids rerunning nearly identical simulations during refinement.
    """

    def __init__(self, load_round_digits: int = 3):
        self.load_round_digits = load_round_digits
        self.cache: Dict[Tuple[int, float], Dict[str, Any]] = {}

    def get(
        self,
        dc_index: int,
        load_mw: float,
    ) -> Dict[str, Any] | None:
        key = (dc_index, round(float(load_mw), self.load_round_digits))
        return self.cache.get(key)

    def set(
        self,
        dc_index: int,
        load_mw: float,
        result: Dict[str, Any],
    ) -> None:
        key = (dc_index, round(float(load_mw), self.load_round_digits))
        self.cache[key] = result

    def __len__(self) -> int:
        return len(self.cache)


# ============================================================
# 5. SIMULATION WRAPPER
# ============================================================

def run_one_dc_with_load(
    datacenter: Dict[str, Any],
    load_mw: float,
    dc_index: int,
    run_id: str,
    save_gif: bool,
    verbose: bool = False,
    fast_evaluation: bool = False,
    output_subdir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calls ``run_physical_simulation_for_params``.

    Expected datacenter dict keys:
        name, lat, lon, temp_c, humidity, solar_wm2, wind_speed_m_s, wind_direction

    ``fast_evaluation=True`` skips almost all disk I/O and uses fewer PDE steps.
    """
    name_safe = str(datacenter.get("name", f"dc_{dc_index}"))
    name_safe = (
        name_safe.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )

    if fast_evaluation:
        output_dir = OUTPUT_ROOT / run_id / "_scratch_eval" / f"dc{dc_index}"
    else:
        output_dir = OUTPUT_ROOT / run_id
        if output_subdir:
            output_dir = output_dir / output_subdir
        output_dir = output_dir / f"{dc_index:02d}_{name_safe}_load_{load_mw:.3f}MW"

    if verbose:
        print("calling run_physical_simulation_for_params", load_mw, "MW", "fast" if fast_evaluation else "full")

    result = run_physical_simulation_for_params(
        lat=float(datacenter["lat"]),
        lon=float(datacenter["lon"]),
        load_mw=float(load_mw),
        temp_c=float(datacenter["temp_c"]),
        humidity=float(datacenter["humidity"]),
        solar_wm2=float(datacenter["solar_wm2"]),
        wind_speed_m_s=float(datacenter["wind_speed_m_s"]),
        wind_direction=str(datacenter["wind_direction"]),
        output_dir=output_dir,
        wind_x=datacenter.get("wind_x", None),
        wind_y=datacenter.get("wind_y", None),
        reverse_geocode=bool(datacenter.get("reverse_geocode", True)),
        all_touched=bool(datacenter.get("all_touched", True)),
        use_nearest_if_not_inside=bool(datacenter.get("use_nearest_if_not_inside", True)),
        steps=FAST_PHYSICS_STEPS if fast_evaluation else int(datacenter.get("physics_steps", DEFAULT_STEPS)),
        save_every=FAST_PHYSICS_SAVE_EVERY
        if fast_evaluation
        else int(datacenter.get("physics_save_every", DEFAULT_SAVE_EVERY)),
        save_gif=save_gif and not fast_evaluation,
        return_states=False if fast_evaluation else None,
        write_disk=not fast_evaluation,
        save_plots=not fast_evaluation,
        save_numpy=not fast_evaluation,
        save_metadata_json=not fast_evaluation,
        verbose=verbose,
    )

    return result


def evaluate_load_distribution(
    datacenters: List[Dict[str, Any]],
    loads_mw: np.ndarray,
    cache: SimulationCache,
    run_id: str,
    save_gif: bool = False,
    verbose: bool = False,
    fast_evaluation: bool = False,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Run simulation for all datacenters with the given load distribution.
    """
    results = []

    for i, datacenter in enumerate(datacenters):
        load_mw = float(loads_mw[i])

        cached = cache.get(i, load_mw)

        if cached is not None:
            result = cached
        else:
            result = run_one_dc_with_load(
                datacenter=datacenter,
                load_mw=load_mw,
                dc_index=i,
                run_id=run_id,
                save_gif=save_gif,
                verbose=verbose,
                fast_evaluation=fast_evaluation,
            )
            cache.set(i, load_mw, result)

        results.append(result)

    objective = compute_total_objective(results)

    return objective, results


# ============================================================
# 6. COORDINATE REFINEMENT
# ============================================================

def coordinate_refine_loads(
    datacenters: List[Dict[str, Any]],
    initial_loads: np.ndarray,
    min_loads: np.ndarray,
    max_loads: np.ndarray,
    cache: SimulationCache,
    run_id: str,
    step_sizes: List[float],
    max_passes_per_step: int = 1,
    verbose: bool = True,
    fast_evaluation: bool = True,
) -> Tuple[np.ndarray, float, List[Dict[str, Any]]]:
    """
    Local search by moving load from one datacenter to another.

    This is fast and works well when the number of datacenters is small.
    """
    current_loads = initial_loads.copy()
    target_total_load = float(current_loads.sum())

    current_obj, current_results = evaluate_load_distribution(
        datacenters=datacenters,
        loads_mw=current_loads,
        cache=cache,
        run_id=run_id,
        save_gif=False,
        verbose=False,
        fast_evaluation=fast_evaluation,
    )

    n = len(current_loads)

    for step in step_sizes:
        if verbose:
            print(f"\nRefinement step size: {step:.4f} MW")

        for pass_idx in range(max_passes_per_step):
            improved = False

            for src in range(n):
                for dst in range(n):
                    if src == dst:
                        continue

                    move = min(
                        step,
                        current_loads[src] - min_loads[src],
                        max_loads[dst] - current_loads[dst],
                    )

                    if move <= 1e-12:
                        continue

                    candidate = current_loads.copy()
                    candidate[src] -= move
                    candidate[dst] += move

                    candidate = project_loads_to_bounds(
                        loads=candidate,
                        min_loads=min_loads,
                        max_loads=max_loads,
                        target_total_load=target_total_load,
                    )

                    candidate_obj, candidate_results = evaluate_load_distribution(
                        datacenters=datacenters,
                        loads_mw=candidate,
                        cache=cache,
                        run_id=run_id,
                        save_gif=False,
                        verbose=False,
                        fast_evaluation=fast_evaluation,
                    )

                    if candidate_obj < current_obj:
                        if verbose:
                            print(
                                f"  improved {current_obj:.4f} -> {candidate_obj:.4f}; "
                                f"move {move:.4f} MW from {src} to {dst}"
                            )

                        current_loads = candidate
                        current_obj = candidate_obj
                        current_results = candidate_results
                        improved = True

            if not improved:
                break

    return current_loads, current_obj, current_results


# ============================================================
# 7. LOAD BOUNDS (MW) + REPORTING
# ============================================================


def _as_mw_vector(value: float | np.ndarray | List[float], n: int) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value.astype(float).reshape(-1)
    else:
        arr = np.ones(n, dtype=float) * float(value)
    if arr.shape[0] != n:
        raise ValueError(f"Expected length {n}, got {arr.shape[0]}")
    return arr


def build_per_site_bounds(
    base_loads_mw: np.ndarray,
    extra_total_load_mw: float,
    min_floor_mw: float,
    max_cap_mw: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-site MW bounds consistent with the dashboard:

    - Each site can take up to ``extra_total_load_mw`` additional MW by itself.
    - Floors keep loads in a realistic operating band.
    """
    n = len(base_loads_mw)
    min_loads = np.maximum(min_floor_mw, base_loads_mw * 0.35)
    max_loads = np.minimum(max_cap_mw, base_loads_mw + float(extra_total_load_mw))
    for i in range(n):
        if max_loads[i] < base_loads_mw[i] + 1e-9:
            max_loads[i] = base_loads_mw[i] + float(extra_total_load_mw)
    return min_loads, max_loads


def slim_eval_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    slim = []
    for r in results:
        slim.append(
            {
                "metrics": r["metrics"],
                "site_objective": compute_single_datacenter_objective(r["metrics"]),
            }
        )
    return slim


def slim_global_item(item: Dict[str, Any]) -> Dict[str, Any]:
    loads = item["loads_mw"]
    return {
        "candidate_index": int(item["candidate_index"]),
        "objective": float(item["objective"]),
        "loads_mw": loads.tolist() if isinstance(loads, np.ndarray) else list(loads),
        "sites": slim_eval_results(item["results"]),
    }


def collect_final_run_asset_paths(
    run_id: str,
    datacenters: List[Dict[str, Any]],
    final_results: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Paths relative to ``OUTPUT_ROOT / run_id /`` (same folder as ``report.html``)."""
    rows: List[Dict[str, str]] = []
    root = OUTPUT_ROOT / run_id
    for i, (dc, r) in enumerate(zip(datacenters, final_results)):
        od = Path(r["output_dir"])
        rel_site = od.relative_to(root).as_posix()
        gif_rel = f"{rel_site}/04_heat_plume_animation.gif"
        gif_path = root / gif_rel
        rows.append(
            {
                "name": str(dc.get("name", f"site_{i}")),
                "masks": f"{rel_site}/01_building_masks.png",
                "final": f"{rel_site}/02_final_temperature.png",
                "anomaly": f"{rel_site}/03_final_anomaly.png",
                "gif": gif_rel if gif_path.is_file() else "",
            }
        )
    return rows


def slim_refined_item(item: Dict[str, Any]) -> Dict[str, Any]:
    loads = item["loads_mw"]
    iloads = item["initial_loads_mw"]
    return {
        "initial_rank": int(item["initial_rank"]),
        "initial_objective": float(item["initial_objective"]),
        "initial_loads_mw": iloads.tolist() if isinstance(iloads, np.ndarray) else list(iloads),
        "objective": float(item["objective"]),
        "loads_mw": loads.tolist() if isinstance(loads, np.ndarray) else list(loads),
        "sites": slim_eval_results(item["results"]),
    }


def build_html_report(
    *,
    run_id: str,
    datacenter_names: List[str],
    base_loads_mw: List[float],
    extra_total_mw: float,
    global_rows: List[Dict[str, Any]],
    refined_rows: List[Dict[str, Any]],
    best_loads_mw: List[float],
    best_objective: float,
    best_per_site: List[Dict[str, Any]],
    asset_rows: List[Dict[str, str]],
) -> str:
    """Self-contained HTML (relative image paths next to this file)."""

    def row_cells(vals: List[str]) -> str:
        return "".join(f"<td>{escape(str(v))}</td>" for v in vals)

    g_head = "<tr><th>#</th><th>Objective</th><th>Loads MW</th></tr>"
    g_body = ""
    for r in global_rows[:25]:
        g_body += "<tr>"
        g_body += row_cells(
            [
                r["candidate_index"],
                f"{r['objective']:.4f}",
                ", ".join(f"{x:.3f}" for x in r["loads_mw"]),
            ]
        )
        g_body += "</tr>"

    rf_head = "<tr><th>Rank</th><th>Obj (before)</th><th>Obj (after)</th><th>Loads MW</th></tr>"
    rf_body = ""
    for r in refined_rows:
        rf_body += "<tr>"
        rf_body += row_cells(
            [
                r["initial_rank"],
                f"{r['initial_objective']:.4f}",
                f"{r['objective']:.4f}",
                ", ".join(f"{x:.3f}" for x in r["loads_mw"]),
            ]
        )
        rf_body += "</tr>"

    best_head = "<tr><th>Site</th><th>Base MW</th><th>Optimal MW</th><th>Δ MW</th><th>Max T °C</th><th>Central ΔT</th></tr>"
    best_body = ""
    for i, name in enumerate(datacenter_names):
        m = best_per_site[i]["metrics"]
        bl = best_loads_mw[i]
        b0 = base_loads_mw[i]
        best_body += "<tr>"
        best_body += row_cells(
            [
                name,
                f"{b0:.3f}",
                f"{bl:.3f}",
                f"{bl - b0:+.3f}",
                f"{float(m.get('max_temp_C', 0)):.2f}",
                f"{float(m.get('central_building_anomaly_C', 0)):.2f}",
            ]
        )
        best_body += "</tr>"

    gallery = ""
    for ar in asset_rows:
        name = escape(ar["name"])
        gallery += f"<h3>{name}</h3><div class='grid'>"
        for label, rel in [
            ("Masks", ar.get("masks", "")),
            ("Final T", ar.get("final", "")),
            ("Anomaly", ar.get("anomaly", "")),
        ]:
            if not rel:
                continue
            url = rel.replace("\\", "/")
            gallery += (
                f"<figure><figcaption>{escape(label)}</figcaption>"
                f"<a href='{escape(url)}' target='_blank'>"
                f"<img loading='lazy' src='{escape(url)}' alt='{escape(label)}'/></a></figure>"
            )
        if ar.get("gif"):
            gurl = str(ar["gif"]).replace("\\", "/")
            gallery += (
                f"<figure><figcaption>Heat GIF</figcaption>"
                f"<img src='{escape(gurl)}' alt='gif'/></figure>"
            )
        gallery += "</div>"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Bayes load optimization — {escape(run_id)}</title>
<style>
body {{ font-family: system-ui, sans-serif; background:#0f172a; color:#e2e8f0; margin:0; padding:24px; }}
h1,h2,h3 {{ color:#38bdf8; }}
table {{ border-collapse:collapse; width:100%; margin:12px 0; font-size:13px; }}
th,td {{ border:1px solid #334155; padding:6px 8px; text-align:left; }}
th {{ background:#1e293b; }}
.grid {{ display:flex; flex-wrap:wrap; gap:12px; }}
figure {{ margin:0; background:#020617; padding:8px; border-radius:8px; max-width:420px; }}
img {{ max-width:100%; height:auto; border-radius:4px; }}
pre {{ background:#020617; padding:12px; overflow:auto; border-radius:8px; font-size:12px; }}
</style></head><body>
<h1>Load optimization report</h1>
<p>Run <code>{escape(run_id)}</code> · extra load to place: <b>{extra_total_mw:.3f} MW</b> · best objective: <b>{best_objective:.4f}</b></p>
<h2>Global search (top candidates shown)</h2>
<table><thead>{g_head}</thead><tbody>{g_body}</tbody></table>
<h2>Refinement starting points</h2>
<table><thead>{rf_head}</thead><tbody>{rf_body}</tbody></table>
<h2>Best allocation (MW)</h2>
<table><thead>{best_head}</thead><tbody>{best_body}</tbody></table>
<h2>Best-run imagery</h2>
{gallery}
<h2>Optimal data JSON</h2>
<p>See <code>optimal_data.json</code> in this folder for machine-readable results.</p>
</body></html>"""


def write_optimization_report_bundle(
    run_root: Path,
    run_id: str,
    datacenters: List[Dict[str, Any]],
    base_loads_mw: np.ndarray,
    extra_total_load_mw: float,
    global_slim: List[Dict[str, Any]],
    refined_slim: List[Dict[str, Any]],
    best_loads_mw: np.ndarray,
    best_objective: float,
    best_results: List[Dict[str, Any]],
    final_asset_rel: List[Dict[str, str]],
) -> Tuple[Path, Path]:
    run_root.mkdir(parents=True, exist_ok=True)
    names = [str(dc.get("name", f"site_{i}")) for i, dc in enumerate(datacenters)]
    best_per = slim_eval_results(best_results)
    optimal_payload = {
        "run_id": run_id,
        "extra_total_load_mw": float(extra_total_load_mw),
        "base_loads_mw": base_loads_mw.tolist(),
        "best_loads_mw": best_loads_mw.tolist(),
        "best_objective": float(best_objective),
        "global_search": global_slim,
        "refinement": refined_slim,
        "best_per_site": best_per,
    }
    json_path = run_root / "optimal_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(optimal_payload, f, indent=2, default=str)

    html = build_html_report(
        run_id=run_id,
        datacenter_names=names,
        base_loads_mw=base_loads_mw.tolist(),
        extra_total_mw=float(extra_total_load_mw),
        global_rows=global_slim,
        refined_rows=refined_slim,
        best_loads_mw=best_loads_mw.tolist(),
        best_objective=float(best_objective),
        best_per_site=best_per,
        asset_rows=final_asset_rel,
    )
    html_path = run_root / "report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return html_path, json_path


# ============================================================
# 8. MAIN OPTIMIZER (streaming + classic)
# ============================================================


def optimize_datacenter_loads_iter(
    datacenters: List[Dict[str, Any]],
    *,
    base_loads_mw: np.ndarray,
    extra_total_load_mw: float,
    min_loads_mw: np.ndarray,
    max_loads_mw: np.ndarray,
    run_id: str,
    n_global_candidates: int = N_GLOBAL_CANDIDATES,
    top_k_refine: int = TOP_K_REFINE,
    random_seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> Iterator[Dict[str, Any]]:
    """
    Yields progress/log dicts, then a final ``{"type": "complete", "result": ...}``.

    All loads are in **MW** (IT load into the physics wrapper).
    """
    if len(datacenters) == 0:
        raise ValueError("datacenters list is empty.")

    rng = np.random.default_rng(random_seed)
    n = len(datacenters)

    base_loads = np.asarray(base_loads_mw, dtype=float).reshape(-1)
    min_loads = np.asarray(min_loads_mw, dtype=float).reshape(-1)
    max_loads = np.asarray(max_loads_mw, dtype=float).reshape(-1)

    if base_loads.shape[0] != n or min_loads.shape[0] != n or max_loads.shape[0] != n:
        raise ValueError("base_loads_mw, min_loads_mw, max_loads_mw must match datacenters count.")

    target_total_load = float(base_loads.sum() + extra_total_load_mw)

    if target_total_load > max_loads.sum() + 1e-6:
        raise ValueError(
            f"Infeasible target total load {target_total_load:.3f} MW (max sum {max_loads.sum():.3f} MW)."
        )

    if target_total_load < min_loads.sum() - 1e-6:
        raise ValueError(
            f"Infeasible target total load {target_total_load:.3f} MW (min sum {min_loads.sum():.3f} MW)."
        )

    cache = SimulationCache(load_round_digits=3)

    candidates = generate_candidate_loads(
        n_datacenters=n,
        base_loads=base_loads,
        min_loads=min_loads,
        max_loads=max_loads,
        extra_total_load_mw=extra_total_load_mw,
        n_candidates=n_global_candidates,
        rng=rng,
    )

    n_cand = len(candidates)
    if extra_total_load_mw > 0:
        step_sizes = [extra_total_load_mw / 4.0, extra_total_load_mw / 10.0]
    else:
        step_sizes = [0.25, 0.1]

    refine_ops_est = top_k_refine * len(step_sizes) * max(1, n * (n - 1))
    total_steps_est = n_cand + refine_ops_est

    yield {
        "type": "plan",
        "run_id": run_id,
        "phase": "init",
        "message": f"Bayes search: {n} sites, {n_cand} load candidates, refine top-{top_k_refine}.",
        "candidate_count": n_cand,
        "target_total_mw": target_total_load,
        "steps_total_estimate": int(total_steps_est),
    }

    global_history: List[Dict[str, Any]] = []

    if verbose:
        print(f"\nGLOBAL LOAD SEARCH — {n_cand} candidates, target {target_total_load:.3f} MW total")

    for idx, loads in enumerate(candidates):
        steps_left = int(total_steps_est - idx)
        yield {
            "type": "progress",
            "phase": "global",
            "index": idx + 1,
            "total": n_cand,
            "steps_remaining": max(0, steps_left),
            "message": f"Evaluating global candidate {idx + 1}/{n_cand}…",
        }

        obj, results = evaluate_load_distribution(
            datacenters=datacenters,
            loads_mw=loads,
            cache=cache,
            run_id=run_id,
            save_gif=False,
            verbose=False,
            fast_evaluation=True,
        )

        item = {
            "candidate_index": idx,
            "loads_mw": loads.copy(),
            "objective": obj,
            "results": results,
        }
        global_history.append(item)

        msg = (
            f"Candidate {idx + 1}/{n_cand}: objective={obj:.4f} · loads MW="
            f"{np.round(loads, 3).tolist()}"
        )
        if verbose:
            print(msg)
        yield {"type": "log", "phase": "global", "message": msg}

    global_history = sorted(global_history, key=lambda x: x["objective"])
    top_candidates = global_history[:top_k_refine]

    yield {
        "type": "log",
        "phase": "global",
        "message": f"Global stage done. Best obj so far: {global_history[0]['objective']:.4f}. Starting refinement…",
    }

    refined_history: List[Dict[str, Any]] = []
    refine_step_counter = 0

    for rank, item in enumerate(top_candidates):
        yield {
            "type": "progress",
            "phase": "refine",
            "index": rank + 1,
            "total": len(top_candidates),
            "steps_remaining": int(refine_ops_est - refine_step_counter),
            "message": f"Refining from global rank {rank + 1}/{len(top_candidates)} (obj={item['objective']:.4f})…",
        }

        refined_loads, refined_obj, refined_results = coordinate_refine_loads(
            datacenters=datacenters,
            initial_loads=item["loads_mw"],
            min_loads=min_loads,
            max_loads=max_loads,
            cache=cache,
            run_id=run_id,
            step_sizes=step_sizes,
            max_passes_per_step=1,
            verbose=False,
            fast_evaluation=True,
        )
        refine_step_counter += 1

        refined_history.append(
            {
                "initial_rank": rank,
                "initial_objective": item["objective"],
                "initial_loads_mw": item["loads_mw"],
                "loads_mw": refined_loads,
                "objective": refined_obj,
                "results": refined_results,
            }
        )

        msg = (
            f"Refined rank {rank + 1}: objective {item['objective']:.4f} → {refined_obj:.4f} · "
            f"loads MW={np.round(refined_loads, 3).tolist()}"
        )
        yield {"type": "log", "phase": "refine", "message": msg}

    refined_history = sorted(refined_history, key=lambda x: x["objective"])
    best = refined_history[0]

    if verbose:
        print(f"\nBEST objective={best['objective']:.4f}, loads={np.round(best['loads_mw'], 4)}")

    global_slim = [slim_global_item(x) for x in global_history]
    refined_slim = [slim_refined_item(x) for x in refined_history]

    result = {
        "best_loads_mw": best["loads_mw"],
        "best_objective": best["objective"],
        "best_results": best["results"],
        "global_history": global_history,
        "refined_history": refined_history,
        "global_slim": global_slim,
        "refined_slim": refined_slim,
        "cache_size": len(cache),
        "run_id": run_id,
        "target_total_mw": target_total_load,
        "base_loads_mw": base_loads,
        "extra_total_load_mw": float(extra_total_load_mw),
    }

    yield {"type": "complete", "result": result}


def optimize_datacenter_loads(
    datacenters: List[Dict[str, Any]],
    *,
    base_loads_mw: np.ndarray | None = None,
    base_load_mw: float = BASE_LOAD_MW,
    extra_total_load_mw: float = EXTRA_TOTAL_LOAD_MW,
    min_loads_mw: float | np.ndarray = MIN_LOAD_MW,
    max_loads_mw: float | np.ndarray = MAX_LOAD_MW,
    run_id: str = "optimization_run",
    n_global_candidates: int = N_GLOBAL_CANDIDATES,
    top_k_refine: int = TOP_K_REFINE,
    random_seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Blocking API: consumes ``optimize_datacenter_loads_iter`` and returns the final dict."""
    n = len(datacenters)
    if base_loads_mw is None:
        base_vec = _as_mw_vector(base_load_mw, n)
    else:
        base_vec = np.asarray(base_loads_mw, dtype=float).reshape(-1)
    min_vec = _as_mw_vector(min_loads_mw, n)
    max_vec = _as_mw_vector(max_loads_mw, n)

    final: Dict[str, Any] | None = None
    for event in optimize_datacenter_loads_iter(
        datacenters,
        base_loads_mw=base_vec,
        extra_total_load_mw=extra_total_load_mw,
        min_loads_mw=min_vec,
        max_loads_mw=max_vec,
        run_id=run_id,
        n_global_candidates=n_global_candidates,
        top_k_refine=top_k_refine,
        random_seed=random_seed,
        verbose=verbose,
    ):
        if event.get("type") == "complete":
            final = event["result"]
    assert final is not None
    return final


# ============================================================
# 9. RERUN BEST WITH FULL OUTPUTS
# ============================================================


def rerun_best_with_full_outputs(
    datacenters: List[Dict[str, Any]],
    best_loads_mw: np.ndarray,
    run_id: str,
    save_gif: bool = True,
    verbose: bool = True,
    output_subdir: str = "best_final_run",
) -> List[Dict[str, Any]]:
    """
    After optimization, rerun the best allocation with full PNG/GIF outputs under
    ``OUTPUT_ROOT / run_id / output_subdir / …``.
    """
    final_results: List[Dict[str, Any]] = []

    for i, datacenter in enumerate(datacenters):
        result = run_one_dc_with_load(
            datacenter=datacenter,
            load_mw=float(best_loads_mw[i]),
            dc_index=i,
            run_id=run_id,
            save_gif=save_gif,
            verbose=verbose,
            fast_evaluation=False,
            output_subdir=output_subdir,
        )
        final_results.append(result)

    return final_results


# ============================================================
# 9. EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    datacenters = [
        {
            "name": "Cold windy site",
            "lat": 43.4723,
            "lon": -80.5449,
            "temp_c": 14.0,
            "humidity": 55.0,
            "solar_wm2": 450.0,
            "wind_speed_m_s": 0.9,
            "wind_direction": "E",
        },
        {
            "name": "Moderate site",
            "lat": 43.6532,
            "lon": -79.3832,
            "temp_c": 21.0,
            "humidity": 60.0,
            "solar_wm2": 600.0,
            "wind_speed_m_s": 0.45,
            "wind_direction": "NE",
        },
        {
            "name": "Warm weak wind site",
            "lat": 42.9849,
            "lon": -81.2453,
            "temp_c": 29.0,
            "humidity": 70.0,
            "solar_wm2": 760.0,
            "wind_speed_m_s": 0.25,
            "wind_direction": "S",
        },
    ]

    result = optimize_datacenter_loads(
        datacenters=datacenters,
        base_load_mw=2.0,
        extra_total_load_mw=5.0,
        min_loads_mw=0.5,
        max_loads_mw=9.0,
        run_id="demo_cli",
        n_global_candidates=12,
        top_k_refine=2,
        random_seed=42,
        verbose=True,
    )

    print("\nBest loads MW:")
    for dc, load in zip(datacenters, result["best_loads_mw"]):
        print(f"{dc['name']}: {load:.4f} MW")

    rerun_best_with_full_outputs(
        datacenters=datacenters,
        best_loads_mw=np.asarray(result["best_loads_mw"], dtype=float),
        run_id="demo_cli",
        save_gif=True,
        verbose=True,
    )