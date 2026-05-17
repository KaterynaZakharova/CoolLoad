from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

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
PER_SITE_FLEET_CAP_MW = MAX_LOAD_MW
# Capped (rejected) sites stay within [cap - slack, cap] during BO/refine/report.
AGENT_CAP_HOLD_SLACK_MW = 1.0

# Fewer total expensive physics evals for quick API runs; each eval = one GP observation.
N_GLOBAL_CANDIDATES = 12
TOP_K_REFINE = 2

RANDOM_SEED = 42

# GP + expected improvement (randomized search over latent z).
BO_EI_XI = 0.03
BO_Z_HALF_WIDTH = 3.0
BO_EI_RANDOM_CANDIDATES = 384
BO_GPR_ALPHA = 1e-6
BO_INIT_MIN = 4

# Fast physics during search (full run uses DEFAULT_STEPS from simulation).
FAST_PHYSICS_STEPS = 700
FAST_PHYSICS_SAVE_EVERY = 120

# Bayes objective: per site, central ΔT vs ambient + mean-field ΔT + weighted mean absolute T.
# ``mean_temp_C`` is O(10–40); keep weight modest so it balances °C-level deltas.
OBJECTIVE_WEIGHT_MEAN_TEMP_C = 0.05

# For optimization, keep outputs light.
# After best load is found, you can rerun best case with save_gif=True.
OPTIMIZATION_SAVE_GIF = False

OUTPUT_ROOT = Path("load_optimization_outputs")


def clear_load_optimization_output_dir(root: Path | None = None) -> None:
    """Remove all previous optimization runs (HTML, PNG, GIF, JSON) under ``root``."""
    import shutil

    r = OUTPUT_ROOT if root is None else root
    if not r.exists():
        return
    for child in list(r.iterdir()):
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except OSError:
            pass


# ============================================================
# 2. OBJECTIVE
# ============================================================

# Objective weights for load spreading.
# Main idea:
#   - strongly minimize the worst temperature rise above ambient
#   - penalize imbalance between datacenter deltas
#   - mildly penalize total delta
#   - optionally penalize hot plume area
#
# Increase OBJECTIVE_WEIGHT_WORST_DELTA if optimizer still puts too much
# load into one datacenter.
OBJECTIVE_WEIGHT_WORST_DELTA = 60.0
OBJECTIVE_WEIGHT_SUM_DELTA = 0.2
OBJECTIVE_WEIGHT_DELTA_STD = 40.0
OBJECTIVE_WEIGHT_HOT_AREA = 0.01

# Smooth max parameter.
# Larger = closer to true max.
# Smaller = smoother for Gaussian Process BO.
OBJECTIVE_SMOOTH_MAX_BETA = 8.0

# Optional absolute-temperature safety penalty.
# This prevents choosing a cold site with acceptable delta but high absolute temp.
OBJECTIVE_WEIGHT_ABSOLUTE_TEMP = 0.15


def _get_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept both:
        result = {"metrics": {...}}
    and:
        result = {...}
    """
    return result["metrics"] if "metrics" in result else result


def smooth_max(values: np.ndarray, beta: float = OBJECTIVE_SMOOTH_MAX_BETA) -> float:
    """
    Smooth approximation of max(values).

    Useful for Bayesian optimization because it is less abrupt than raw max().
    """
    values = np.asarray(values, dtype=float).reshape(-1)

    if values.size == 0:
        return 0.0

    m = float(np.max(values))
    return float(m + np.log(np.sum(np.exp(beta * (values - m)))) / beta)


def compute_single_datacenter_objective(metrics: Dict[str, Any]) -> float:
    """
    Per-site diagnostic objective.

    This is mostly used for reporting in slim_eval_results().
    The true allocation objective is compute_total_objective(), because load
    spreading requires comparing all datacenters together.
    """
    central_delta = float(metrics.get("central_building_anomaly_C", 0.0))
    max_delta = float(metrics.get("max_anomaly_C", central_delta))
    mean_delta = float(metrics.get("mean_anomaly_C", 0.0))

    hot1 = float(metrics.get("hot_area_gt_1C_cells", 0.0))
    hot2 = float(metrics.get("hot_area_gt_2C_cells", 0.0))
    hot5 = float(metrics.get("hot_area_gt_5C_cells", 0.0))

    hot_area_penalty = hot1 + 3.0 * hot2 + 10.0 * hot5

    return float(
        2.0 * max_delta
        + 1.0 * central_delta
        + 0.5 * mean_delta
        + OBJECTIVE_WEIGHT_HOT_AREA * hot_area_penalty
    )


def compute_total_objective(
    results: List[Dict[str, Any]],
    site_weights: Optional[np.ndarray] = None,
) -> float:
    """
    Load-spreading objective for Bayesian optimization.

    Lower is better.

    This objective is designed to avoid the trivial solution:
        "put everything into the single coldest/best datacenter."

    It minimizes:
        1. worst temperature delta across datacenters,
        2. imbalance of temperature deltas,
        3. total temperature delta,
        4. hot plume area,
        5. mild absolute-temperature risk.

    The most important term is the smooth worst-delta term.
    """
    if len(results) == 0:
        return 0.0

    max_deltas = []
    central_deltas = []
    mean_deltas = []
    max_temps = []
    hot_area_penalty = 0.0
    n = len(results)
    weights = np.ones(n, dtype=float)
    if site_weights is not None:
        weights = np.asarray(site_weights, dtype=float).reshape(-1)
        if weights.size != n:
            raise ValueError("site_weights length must match results length")

    for i, result in enumerate(results):
        metrics = _get_metrics(result)
        w = float(weights[i])

        central_raw = float(metrics.get("central_building_anomaly_C", 0.0))
        max_raw = float(metrics.get("max_anomaly_C", central_raw))
        central_delta = central_raw * w
        max_delta = max_raw * w
        mean_delta = float(metrics.get("mean_anomaly_C", 0.0)) * w
        max_temp = float(metrics.get("max_temp_C", 0.0))

        hot1 = float(metrics.get("hot_area_gt_1C_cells", 0.0))
        hot2 = float(metrics.get("hot_area_gt_2C_cells", 0.0))
        hot5 = float(metrics.get("hot_area_gt_5C_cells", 0.0))

        max_deltas.append(max_delta)
        central_deltas.append(central_delta)
        mean_deltas.append(mean_delta)
        max_temps.append(max_temp)

        hot_area_penalty += w * (hot1 + 3.0 * hot2 + 10.0 * hot5)

    max_deltas = np.asarray(max_deltas, dtype=float)
    central_deltas = np.asarray(central_deltas, dtype=float)
    mean_deltas = np.asarray(mean_deltas, dtype=float)
    max_temps = np.asarray(max_temps, dtype=float)

    # Main spreading metric:
    # if one datacenter gets overloaded, its delta rises and this term grows fast.
    worst_delta_smooth = smooth_max(max_deltas)

    # Imbalance terms:
    # make the optimizer prefer similar thermal stress across sites.
    max_delta_std = float(np.std(max_deltas))
    central_delta_std = float(np.std(central_deltas))

    # Total heating terms:
    # keep the overall plume small, but do not let this dominate.
    sum_max_delta = float(np.sum(max_deltas))
    sum_mean_delta = float(np.sum(mean_deltas))

    # Mild absolute thermal safety:
    # keeps very hot absolute temperature from being ignored.
    worst_abs_temp = smooth_max(max_temps)

    objective = (
        OBJECTIVE_WEIGHT_WORST_DELTA * worst_delta_smooth
        + OBJECTIVE_WEIGHT_SUM_DELTA * sum_max_delta
        + OBJECTIVE_WEIGHT_DELTA_STD * (max_delta_std + central_delta_std)
        + OBJECTIVE_WEIGHT_HOT_AREA * hot_area_penalty
        + OBJECTIVE_WEIGHT_ABSOLUTE_TEMP * worst_abs_temp
        + 0.25 * sum_mean_delta
    )

    return float(objective)


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


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x - float(np.max(x))
    ex = np.exp(np.clip(x, -40.0, 40.0))
    s = float(np.sum(ex))
    return ex / max(s, 1e-300)


def decode_latent_z_to_loads_mw(
    z: np.ndarray,
    base_loads: np.ndarray,
    min_loads: np.ndarray,
    max_loads: np.ndarray,
    extra_total_load_mw: float,
    target_total_load: float,
) -> np.ndarray:
    """
    Map unconstrained latent ``z`` (length n-1) to feasible MW loads.

    Uses a softmax on ``n`` logits (z padded with 0) to split ``extra_total_load_mw``
    across sites, then ``project_loads_to_bounds`` to enforce per-site min/max and total.
    """
    base_loads = np.asarray(base_loads, dtype=float).reshape(-1)
    min_loads = np.asarray(min_loads, dtype=float).reshape(-1)
    max_loads = np.asarray(max_loads, dtype=float).reshape(-1)
    n = int(base_loads.size)
    extra = float(extra_total_load_mw)

    if n == 1:
        loads = np.array([target_total_load], dtype=float)
        return project_loads_to_bounds(loads, min_loads, max_loads, target_total_load)

    z = np.asarray(z, dtype=float).reshape(-1)
    if z.size != n - 1:
        raise ValueError(f"z must have length {n - 1}, got {z.size}")

    logits = np.concatenate([z, np.zeros(1, dtype=float)])
    p = _softmax(logits)
    loads = base_loads + p * extra
    return project_loads_to_bounds(loads, min_loads, max_loads, target_total_load)


def expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    y_best: float,
    xi: float = BO_EI_XI,
) -> np.ndarray:
    """Gaussian-process expected improvement (batch)."""
    mu = np.asarray(mu, dtype=float).reshape(-1)
    sigma = np.maximum(np.asarray(sigma, dtype=float).reshape(-1), 1e-9)
    imp = y_best - mu - xi
    with np.errstate(divide="ignore", invalid="ignore"):
        z = imp / sigma
    ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _fit_gp(
    X: np.ndarray,
    y: np.ndarray,
    random_seed: int,
) -> GaussianProcessRegressor | None:
    """Fit GPR on latent points; returns None if not enough data."""
    if X.shape[0] < 2 or X.shape[1] < 1:
        return None
    d = X.shape[1]
    ls = 1.0 if d == 1 else np.ones(d)
    kernel = Matern(length_scale=ls, nu=2.5) + WhiteKernel(noise_level=1e-4)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=BO_GPR_ALPHA,
        normalize_y=True,
        random_state=random_seed,
        n_restarts_optimizer=0,
    )
    gp.fit(X, y)
    return gp


def _suggest_next_z_ei(
    gp: GaussianProcessRegressor,
    y_obs: np.ndarray,
    rng: np.random.Generator,
    d: int,
    n_candidates: int = BO_EI_RANDOM_CANDIDATES,
) -> np.ndarray:
    """Random search for z maximizing expected improvement."""
    y_best = float(np.min(y_obs))
    box = BO_Z_HALF_WIDTH
    Zcand = rng.uniform(-box, box, size=(n_candidates, d))
    mu, std = gp.predict(Zcand, return_std=True)
    ei = expected_improvement(mu, std, y_best)
    j = int(np.argmax(ei))
    return Zcand[j].copy()


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
    save_numpy: Optional[bool] = None,
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

    if save_numpy is None:
        save_numpy_flag = not fast_evaluation
    else:
        save_numpy_flag = bool(save_numpy) and not fast_evaluation

    sim_kwargs: Dict[str, Any] = {}
    if datacenter.get("datacenter_specs") is not None:
        sim_kwargs["datacenter_specs"] = datacenter["datacenter_specs"]
    elif datacenter.get("wall_specific_heat_kj_per_kg_k") is not None:
        sim_kwargs["wall_specific_heat_kj_per_kg_k"] = float(datacenter["wall_specific_heat_kj_per_kg_k"])
        if datacenter.get("wall_density_kg_m3") is not None:
            sim_kwargs["wall_density_kg_m3"] = float(datacenter["wall_density_kg_m3"])
        if datacenter.get("wall_material") is not None:
            sim_kwargs["wall_material"] = str(datacenter["wall_material"])

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
        save_numpy=save_numpy_flag,
        save_metadata_json=not fast_evaluation,
        verbose=verbose,
        **sim_kwargs,
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
    objective_context: Optional[Dict[str, Any]] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Run simulation for all datacenters with the given load distribution.

    ``objective_context`` may include:
        - ``site_weights``: per-site multipliers on thermal terms (orchestrator / agents)
        - ``max_loads_mw``: hard per-site MW caps before evaluation
    """
    loads_mw = np.asarray(loads_mw, dtype=float).reshape(-1).copy()
    n_dc = len(datacenters)
    if objective_context and objective_context.get("min_loads_mw") is not None:
        min_for_clip = np.asarray(objective_context["min_loads_mw"], dtype=float).reshape(-1)
    else:
        min_for_clip = np.full(n_dc, MIN_LOAD_MW, dtype=float)
    loads_mw = clip_loads_for_objective_context(
        loads_mw, min_for_clip, objective_context
    )

    site_weights = None
    if objective_context and objective_context.get("site_weights") is not None:
        site_weights = np.asarray(objective_context["site_weights"], dtype=float)

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

    objective = compute_total_objective(results, site_weights=site_weights)

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
    objective_context: Optional[Dict[str, Any]] = None,
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
        objective_context=objective_context,
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
                        objective_context=objective_context,
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
    Per-site MW bounds consistent with the dashboard.

    Each site's load is at least ``max(base, min_floor_mw)``. Each site may go up to
    ``max_cap_mw`` (default 150 MW per datacenter). Fleet-wide headroom is up to
    ``max_cap_mw * n_sites`` — we do **not** tighten per-site max from the target split,
    so high baselines + moderate extra stay feasible.
    """
    base = np.asarray(base_loads_mw, dtype=float).reshape(-1)
    n = int(base.size)
    extra = float(extra_total_load_mw)
    per_site_cap = float(max_cap_mw)
    floor_f = float(min_floor_mw)

    target_total = float(base.sum() + extra)
    min_loads = np.maximum(base, floor_f)
    sum_min = float(min_loads.sum())

    # If mins alone exceed target (e.g. negative extra), allow mins down to baseline.
    if sum_min > target_total + 1e-6:
        min_loads = base.copy()
        sum_min = float(min_loads.sum())

    max_loads = np.maximum(min_loads, per_site_cap)

    sum_max = float(max_loads.sum())
    if sum_max < target_total - 1e-6:
        raise ValueError(
            f"Infeasible target total load {target_total:.3f} MW: "
            f"fleet max is {sum_max:.3f} MW ({per_site_cap:.0f} MW × {n} sites). "
            f"Raise max_cap_mw or reduce extra_total_load_mw."
        )
    if float(min_loads.sum()) > target_total + 1e-6:
        raise ValueError(
            f"Infeasible target total load {target_total:.3f} MW: "
            f"minimum sum {float(min_loads.sum()):.3f} MW (baselines). "
            f"Increase extra_total_load_mw or lower base loads."
        )

    return min_loads, max_loads


def merge_objective_max_loads(
    min_loads: np.ndarray,
    max_loads: np.ndarray,
    objective_context: Optional[Dict[str, Any]],
    fleet_cap_mw: float = PER_SITE_FLEET_CAP_MW,
) -> np.ndarray:
    """
    Keep per-site fleet headroom (``fleet_cap_mw``) on accepted sites.

    Only sites listed in ``rejected_site_indices`` (or with caps below fleet)
    are tightened from ``objective_context['max_loads_mw']``.
    """
    min_loads = np.asarray(min_loads, dtype=float).reshape(-1)
    max_loads = np.asarray(max_loads, dtype=float).reshape(-1).copy()
    n = int(max_loads.size)
    fleet_cap = float(
        (objective_context or {}).get("per_site_fleet_cap_mw", fleet_cap_mw)
    )
    fleet_max = np.maximum(min_loads, fleet_cap)

    if not objective_context or objective_context.get("max_loads_mw") is None:
        return np.maximum(max_loads, fleet_max)

    caps = np.asarray(objective_context["max_loads_mw"], dtype=float).reshape(-1)
    if caps.size != n:
        return np.maximum(max_loads, fleet_max)

    rejected = set(objective_context.get("rejected_site_indices") or [])
    if not rejected:
        for i in range(n):
            if caps[i] < fleet_max[i] - 1e-3:
                rejected.add(i)

    out = fleet_max.copy()
    for i in rejected:
        out[i] = min(fleet_max[i], max(min_loads[i], caps[i]))
    return out


def clip_loads_for_objective_context(
    loads_mw: np.ndarray,
    min_loads: np.ndarray,
    objective_context: Optional[Dict[str, Any]],
    fleet_cap_mw: float = PER_SITE_FLEET_CAP_MW,
) -> np.ndarray:
    """Clip candidate loads only on rejected sites (orchestrator caps)."""
    if not objective_context or objective_context.get("max_loads_mw") is None:
        return np.asarray(loads_mw, dtype=float).reshape(-1).copy()

    min_loads = np.asarray(min_loads, dtype=float).reshape(-1)
    loads = np.asarray(loads_mw, dtype=float).reshape(-1).copy()
    n = loads.size
    caps = np.asarray(objective_context["max_loads_mw"], dtype=float).reshape(-1)
    if caps.size != n:
        return loads

    fleet_cap = float(objective_context.get("per_site_fleet_cap_mw", fleet_cap_mw))
    fleet_max = np.maximum(min_loads, fleet_cap)
    rejected = set(objective_context.get("rejected_site_indices") or [])
    if not rejected:
        for i in range(n):
            if caps[i] < fleet_max[i] - 1e-3:
                rejected.add(i)

    for i in rejected:
        loads[i] = min(loads[i], max(min_loads[i], caps[i]))
    return loads


def _rejected_site_indices_from_context(
    objective_context: Optional[Dict[str, Any]],
    n: int,
    fleet_max: np.ndarray,
    caps: np.ndarray,
) -> set[int]:
    rejected = set(objective_context.get("rejected_site_indices") or []) if objective_context else set()
    if not rejected and objective_context and caps.size == n:
        for i in range(n):
            if caps[i] < fleet_max[i] - 1e-3:
                rejected.add(i)
    return rejected


def resolve_agent_load_bounds(
    min_loads: np.ndarray,
    max_loads: np.ndarray,
    objective_context: Optional[Dict[str, Any]],
    fleet_cap_mw: float = PER_SITE_FLEET_CAP_MW,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge orchestrator MW caps into max bounds and raise mins on capped sites so
    refinement cannot drain a rejected site to baseline when the cap is ~81 MW.
    """
    min_loads = np.asarray(min_loads, dtype=float).reshape(-1).copy()
    max_loads = np.asarray(max_loads, dtype=float).reshape(-1).copy()
    n = int(max_loads.size)
    max_loads = merge_objective_max_loads(min_loads, max_loads, objective_context, fleet_cap_mw)

    if not objective_context or objective_context.get("max_loads_mw") is None:
        return min_loads, max_loads

    caps = np.asarray(objective_context["max_loads_mw"], dtype=float).reshape(-1)
    if caps.size != n:
        return min_loads, max_loads

    fleet_cap = float(objective_context.get("per_site_fleet_cap_mw", fleet_cap_mw))
    fleet_max = np.maximum(min_loads, fleet_cap)
    rejected = _rejected_site_indices_from_context(objective_context, n, fleet_max, caps)
    slack = float(objective_context.get("agent_cap_hold_slack_mw", AGENT_CAP_HOLD_SLACK_MW))

    for i in rejected:
        cap_i = float(max_loads[i])
        floor_i = max(float(min_loads[i]), cap_i - slack)
        min_loads[i] = min(floor_i, cap_i)

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
    slim: Dict[str, Any] = {
        "candidate_index": int(item["candidate_index"]),
        "objective": float(item["objective"]),
        "loads_mw": loads.tolist() if isinstance(loads, np.ndarray) else list(loads),
        "sites": slim_eval_results(item["results"]),
    }
    lz = item.get("latent_z")
    if lz is not None:
        slim["latent_z"] = lz.tolist() if isinstance(lz, np.ndarray) else list(lz)
    return slim


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


def write_optimization_charts(
    run_root: Path,
    global_slim: List[Dict[str, Any]],
    refined_slim: List[Dict[str, Any]],
    datacenter_names: List[str],
    base_loads_mw: List[float],
    best_loads_mw: List[float],
    random_seed: int = RANDOM_SEED,
    best_per_site: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, str]:
    """
    Save PNG charts next to ``report.html``.

    Includes:
        - Objective vs evaluation step with running best (transparency on improvement).
        - GP surrogate mean ± σ at training latents (exploration uncertainty proxy).
        - Per-site load distribution across candidates (p10–p90 band vs best).
        - Baseline vs optimal bars with σ across candidates.
        - Heatmap of MW splits for the lowest-objective trials (readability).
        - Per-site diagnostic thermal score at the optimum.
        - Local refinement as a single before→after path chart per seed.
    """
    run_root.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    if not global_slim:
        return paths

    base_arr = np.asarray(base_loads_mw, dtype=float).reshape(-1)
    best_arr = np.asarray(best_loads_mw, dtype=float).reshape(-1)
    n_sites = len(datacenter_names)

    rows_ts = sorted(global_slim, key=lambda r: int(r["candidate_index"]))
    steps = np.array([int(r["candidate_index"]) + 1 for r in rows_ts], dtype=float)
    objs_ts = np.array([float(r["objective"]) for r in rows_ts], dtype=float)
    cum_best = np.minimum.accumulate(objs_ts)
    regret = objs_ts - cum_best

    loads_mat = np.array([row["loads_mw"] for row in global_slim], dtype=float)
    load_std = np.std(loads_mat, axis=0) if loads_mat.size else np.zeros(n_sites)
    load_p10 = np.percentile(loads_mat, 10, axis=0) if loads_mat.size else best_arr
    load_p90 = np.percentile(loads_mat, 90, axis=0) if loads_mat.size else best_arr

    def _style_dark(ax: Any) -> None:
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # --- 1) Improvement trace (objective + running best + regret band) ---
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    ax.plot(
        steps,
        objs_ts,
        "o-",
        color="#94a3b8",
        lw=1.4,
        ms=5,
        label="Objective each physics eval (lower is better)",
        alpha=0.9,
    )
    ax.plot(
        steps,
        cum_best,
        "-",
        color="#22d3ee",
        lw=2.8,
        label="Best-so-far (running minimum)",
    )
    ax.fill_between(
        steps,
        cum_best,
        objs_ts,
        where=(objs_ts >= cum_best),
        color="#f97316",
        alpha=0.18,
        interpolate=True,
        label="Instant regret (gap to best-so-far)",
    )
    ax.set_xlabel("Physics evaluation # (chronological)", color="#94a3b8", fontsize=11)
    ax.set_ylabel("Objective", color="#94a3b8", fontsize=11)
    ax.set_title(
        "Search progress: how fast did we improve?",
        color="#e2e8f0",
        fontsize=13,
        fontweight="semibold",
    )
    _style_dark(ax)
    leg = ax.legend(loc="upper right", fontsize=9, facecolor="#1e293b", edgecolor="#334155")
    for t in leg.get_texts():
        t.set_color("#e2e8f0")
    fig.tight_layout()
    p_trace = run_root / "chart_improvement_trace.png"
    fig.savefig(p_trace, dpi=165, facecolor=fig.get_facecolor())
    plt.close(fig)
    paths["improvement_trace"] = p_trace.name

    # --- 2) GP mean ± σ at evaluated latents (final surrogate, training-only) ---
    Z_list = [r.get("latent_z") for r in rows_ts]
    if (
        Z_list
        and all(z is not None and len(z) > 0 for z in Z_list)
        and len(Z_list) >= 2
    ):
        X_train = np.array([np.asarray(z, dtype=float).reshape(-1) for z in Z_list], dtype=float)
        y_train = objs_ts.copy()
        gp_final = _fit_gp(X_train, y_train, random_seed)
        if gp_final is not None:
            mu_t, std_t = gp_final.predict(X_train, return_std=True)
            fig, ax = plt.subplots(figsize=(10, 5.0), facecolor="#0f172a")
            ax.set_facecolor("#0f172a")
            ax.plot(steps, y_train, "o", color="#fbbf24", ms=6, label="Observed objective")
            ax.plot(steps, mu_t, "-", color="#a78bfa", lw=2.2, label="GP posterior mean μ(z)")
            ax.fill_between(
                steps,
                mu_t - std_t,
                mu_t + std_t,
                color="#a78bfa",
                alpha=0.25,
                label="±1σ GP uncertainty (surrogate, not physics noise)",
            )
            ax.set_xlabel("Evaluation #", color="#94a3b8", fontsize=11)
            ax.set_ylabel("Objective", color="#94a3b8", fontsize=11)
            ax.set_title(
                "Surrogate model (GPR) vs observations — where the algorithm is uncertain",
                color="#e2e8f0",
                fontsize=13,
                fontweight="semibold",
            )
            _style_dark(ax)
            leg = ax.legend(loc="upper right", facecolor="#1e293b", edgecolor="#334155", fontsize=9)
            for t in leg.get_texts():
                t.set_color("#e2e8f0")
            fig.tight_layout()
            p_gp = run_root / "chart_gp_surrogate.png"
            fig.savefig(p_gp, dpi=165, facecolor=fig.get_facecolor())
            plt.close(fig)
            paths["gp_surrogate"] = p_gp.name

    # --- 3) Per-site MW: baseline, best, p10–p90 across candidate pool ---
    fig, ax = plt.subplots(figsize=(max(9.0, 1.2 * n_sites), 5.0), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    x_pos = np.arange(n_sites)
    w = 0.28
    ax.bar(x_pos - w, base_arr, w, label="Baseline MW", color="#475569", edgecolor="#64748b")
    ax.bar(x_pos, best_arr, w, label="Optimized MW", color="#22d3ee", edgecolor="#0891b2")
    ax.errorbar(
        x_pos,
        best_arr,
        yerr=[np.maximum(0, best_arr - load_p10), np.maximum(0, load_p90 - best_arr)],
        fmt="none",
        ecolor="#fb923c",
        capsize=4,
        elinewidth=1.5,
        label="p10–p90 MW across evaluated splits (at optimal x)",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [n[:18] + ("…" if len(n) > 18 else "") for n in datacenter_names],
        rotation=28,
        ha="right",
        color="#94a3b8",
        fontsize=9,
    )
    ax.set_ylabel("MW", color="#94a3b8", fontsize=11)
    ax.set_title(
        "Load allocation: baseline vs optimal vs cross-candidate spread",
        color="#e2e8f0",
        fontsize=13,
        fontweight="semibold",
    )
    _style_dark(ax)
    leg = ax.legend(loc="upper left", facecolor="#1e293b", edgecolor="#334155", fontsize=9)
    for t in leg.get_texts():
        t.set_color("#e2e8f0")
    fig.tight_layout()
    p2 = run_root / "chart_loads.png"
    fig.savefig(p2, dpi=165, facecolor=fig.get_facecolor())
    plt.close(fig)
    paths["loads"] = p2.name

    # --- 4) Heatmap: MW split for lowest-objective trials (which site got load?) ---
    rows_by_obj = sorted(global_slim, key=lambda r: float(r["objective"]))
    n_hm = min(24, len(rows_by_obj))
    if n_hm >= 1 and n_sites >= 1:
        hm_rows = rows_by_obj[:n_hm]
        mat = np.array([list(map(float, r["loads_mw"])) for r in hm_rows], dtype=float)
        fig_h, ax_h = plt.subplots(figsize=(max(8.0, 0.55 * n_sites + 3.0), max(4.5, 0.32 * n_hm + 2.2)), facecolor="#0f172a")
        ax_h.set_facecolor("#0f172a")
        im_h = ax_h.imshow(mat, aspect="auto", cmap="magma", interpolation="nearest")
        cb = fig_h.colorbar(im_h, ax=ax_h, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color="#94a3b8")
        cb.set_label("MW at site", color="#94a3b8", fontsize=10)
        plt.setp(cb.ax.get_yticklabels(), color="#94a3b8")
        ax_h.set_xticks(np.arange(n_sites))
        ax_h.set_xticklabels(
            [n[:14] + ("…" if len(n) > 14 else "") for n in datacenter_names],
            rotation=35,
            ha="right",
            color="#94a3b8",
            fontsize=9,
        )
        y_labs = [
            f"#{int(r['candidate_index']) + 1}  obj={float(r['objective']):.3f}"
            for r in hm_rows
        ]
        ax_h.set_yticks(np.arange(n_hm))
        ax_h.set_yticklabels(y_labs, fontsize=8, color="#94a3b8")
        ax_h.set_title(
            "Load splits for the lowest-objective trials (rows ordered best → worse in this window)",
            color="#e2e8f0",
            fontsize=12,
            fontweight="semibold",
        )
        _style_dark(ax_h)
        fig_h.tight_layout()
        p_hm = run_root / "chart_load_heatmap.png"
        fig_h.savefig(p_hm, dpi=165, facecolor=fig_h.get_facecolor())
        plt.close(fig_h)
        paths["load_heatmap"] = p_hm.name

    # --- 5) ΔMW per site (improvement direction) ---
    delta = best_arr - base_arr
    fig, ax = plt.subplots(figsize=(max(8.0, 1.0 * n_sites), 4.4), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    colors = ["#34d399" if d >= 0 else "#f87171" for d in delta]
    ax.bar(x_pos, delta, color=colors, edgecolor="#334155")
    ax.axhline(0, color="#94a3b8", lw=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [n[:18] + ("…" if len(n) > 18 else "") for n in datacenter_names],
        rotation=28,
        ha="right",
        color="#94a3b8",
        fontsize=9,
    )
    ax.set_ylabel("Δ MW (optimal − baseline)", color="#94a3b8", fontsize=11)
    ax.set_title("Extra load absorbed per site at optimum", color="#e2e8f0", fontsize=13, fontweight="semibold")
    _style_dark(ax)
    fig.tight_layout()
    p_delta = run_root / "chart_delta_mw.png"
    fig.savefig(p_delta, dpi=165, facecolor=fig.get_facecolor())
    plt.close(fig)
    paths["delta_mw"] = p_delta.name

    # --- 6) Per-site diagnostic score at optimum (same formula as site_objective in JSON) ---
    if best_per_site and len(best_per_site) == n_sites and n_sites > 0:
        site_obj = [float(r.get("site_objective", 0.0)) for r in best_per_site]
        fig_s, ax_s = plt.subplots(figsize=(max(8.0, 0.45 * n_sites + 4.0), 4.8), facecolor="#0f172a")
        ax_s.set_facecolor("#0f172a")
        y_pos = np.arange(n_sites)
        ax_s.barh(y_pos, site_obj, color="#38bdf8", edgecolor="#0891b2", height=0.65)
        ax_s.set_yticks(y_pos)
        ax_s.set_yticklabels(
            [n[:22] + ("…" if len(n) > 22 else "") for n in datacenter_names],
            color="#94a3b8",
            fontsize=9,
        )
        ax_s.invert_yaxis()
        ax_s.set_xlabel(
            "Per-site diagnostic thermal score (lower = calmer site; not equal to global objective)",
            color="#94a3b8",
            fontsize=10,
        )
        ax_s.set_title(
            "Thermal stress by site at the chosen optimum",
            color="#e2e8f0",
            fontsize=13,
            fontweight="semibold",
        )
        _style_dark(ax_s)
        fig_s.tight_layout()
        p_site = run_root / "chart_site_scores.png"
        fig_s.savefig(p_site, dpi=165, facecolor=fig_s.get_facecolor())
        plt.close(fig_s)
        paths["site_scores"] = p_site.name

    # --- 7) Local refinement: one before→after connector per seed ---
    if refined_slim:
        fig, ax = plt.subplots(figsize=(9.0, max(3.8, 0.55 * len(refined_slim))), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        before = [float(r["initial_objective"]) for r in refined_slim]
        after = [float(r["objective"]) for r in refined_slim]
        y = np.arange(len(refined_slim), dtype=float)
        for i, yi in enumerate(y):
            b, a = before[i], after[i]
            ax.plot([b, a], [yi, yi], "-", color="#475569", lw=2.4, zorder=1)
            ax.scatter([b], [yi], s=96, c="#94a3b8", edgecolors="#e2e8f0", linewidths=0.6, zorder=3)
            ax.scatter([a], [yi], s=110, c="#34d399", edgecolors="#059669", linewidths=0.6, zorder=4)
            if a < b - 1e-9:
                ax.annotate(
                    f"Δ {a - b:.3f}",
                    xy=(a, yi),
                    xytext=(6, 0),
                    textcoords="offset points",
                    fontsize=8,
                    color="#6ee7b7",
                    va="center",
                )
        ax.set_yticks(y)
        ax.set_yticklabels(
            [f"Seed (GP rank {int(r['initial_rank']) + 1})" for r in refined_slim],
            color="#94a3b8",
            fontsize=10,
        )
        ax.set_xlabel("Objective (lower is better)", color="#94a3b8", fontsize=11)
        ax.set_title(
            "Local coordinate refinement: gray = before, green = after",
            color="#e2e8f0",
            fontsize=13,
            fontweight="semibold",
        )
        leg_el = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#94a3b8", markersize=9, label="Before refine"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#34d399", markersize=10, label="After refine"),
        ]
        leg = ax.legend(handles=leg_el, loc="lower right", facecolor="#1e293b", edgecolor="#334155")
        for t in leg.get_texts():
            t.set_color("#e2e8f0")
        _style_dark(ax)
        fig.tight_layout()
        p3 = run_root / "chart_refinement.png"
        fig.savefig(p3, dpi=165, facecolor=fig.get_facecolor())
        plt.close(fig)
        paths["refinement"] = p3.name

    return paths


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
    chart_paths: Dict[str, str],
    narrative_html: Optional[str] = None,
    effective_min_loads_mw: Optional[List[float]] = None,
    effective_max_loads_mw: Optional[List[float]] = None,
) -> str:
    """HTML report with chart and gallery URLs rooted at ``/opt-out/{run_id}/`` for HTTP serving."""

    def row_cells(vals: List[str]) -> str:
        return "".join(f"<td>{escape(str(v))}</td>" for v in vals)

    g_head = "<tr><th>#</th><th>Objective</th><th>Loads MW</th></tr>"
    g_body = ""
    for r in global_rows[:40]:
        g_body += "<tr>"
        g_body += row_cells(
            [
                r["candidate_index"],
                f"{r['objective']:.4f}",
                ", ".join(f"{x:.3f}" for x in r["loads_mw"]),
            ]
        )
        g_body += "</tr>"

    rf_head = "<tr><th>Seed</th><th>Obj (before)</th><th>Obj (after)</th><th>Loads MW</th></tr>"
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

    has_caps = (
        effective_max_loads_mw is not None
        and len(effective_max_loads_mw) == len(datacenter_names)
    )
    cap_head = "<th>Agent cap MW</th>" if has_caps else ""
    best_head = (
        f"<tr><th>Site</th><th>Base MW</th>{cap_head}"
        "<th>Optimal MW</th><th>Δ MW</th><th>Max T °C</th><th>Central ΔT</th></tr>"
    )
    best_body = ""
    for i, name in enumerate(datacenter_names):
        m = best_per_site[i]["metrics"]
        bl = best_loads_mw[i]
        b0 = base_loads_mw[i]
        cap_cell = ""
        if has_caps:
            cap_v = float(effective_max_loads_mw[i])
            fleet = float(PER_SITE_FLEET_CAP_MW)
            cap_cell = (
                f"<td>{cap_v:.3f}</td>"
                if cap_v < fleet - 1e-3
                else "<td>—</td>"
            )
        best_body += "<tr>"
        best_body += f"<td>{escape(name)}</td><td>{b0:.3f}</td>{cap_cell}"
        best_body += row_cells(
            [
                f"{bl:.3f}",
                f"{bl - b0:+.3f}",
                f"{float(m.get('max_temp_C', 0)):.2f}",
                f"{float(m.get('central_building_anomaly_C', 0)):.2f}",
            ]
        )
        best_body += "</tr>"

    def pub_url(rel: str) -> str:
        if not rel:
            return ""
        r = rel.replace("\\", "/").lstrip("./")
        if r.startswith("/opt-out/") or r.startswith("http://") or r.startswith("https://"):
            return r
        return f"/opt-out/{run_id}/{r}".replace("//", "/")

    def img_block(rel: str, caption: str) -> str:
        if not rel:
            return ""
        u = pub_url(rel)
        return (
            f"<figure class='chart'><figcaption>{escape(caption)}</figcaption>"
            f"<img src='{escape(u)}' alt='{escape(caption)}'/></figure>"
        )

    charts_html = "<div class='charts'>"
    charts_html += "<h2>Optimization at a glance</h2>"
    charts_html += f"<p class='lead'>Figure URLs use <code>/opt-out/{escape(run_id)}/…</code> so they resolve when this report is served from the API (or any reverse proxy that maps that prefix). "
    charts_html += "Chronological panels show each expensive physics evaluation; orange regret bands show how far a point was from the running best. "
    charts_html += "The GP chart is a <em>surrogate</em> on latent load codes (±1σ there is model uncertainty in z-space, not a full simulator credible interval). "
    charts_html += "The heatmap contrasts <strong>which sites received MW</strong> in the best-scoring trials.</p>"
    charts_html += "<div class='chart-row'>"
    charts_html += img_block(chart_paths.get("improvement_trace", ""), "Objective vs evaluation # (running best + regret)")
    charts_html += img_block(chart_paths.get("gp_surrogate", ""), "GPR surrogate vs observations (mean ± σ)")
    charts_html += "</div><div class='chart-row'>"
    charts_html += img_block(chart_paths.get("loads", ""), "Baseline vs optimal MW + cross-candidate spread")
    charts_html += img_block(chart_paths.get("load_heatmap", ""), "MW split heatmap for lowest-objective trials")
    charts_html += "</div><div class='chart-row'>"
    charts_html += img_block(chart_paths.get("delta_mw", ""), "Δ MW per site (optimal − baseline)")
    charts_html += img_block(chart_paths.get("site_scores", ""), "Per-site diagnostic thermal score at optimum")
    charts_html += "</div><div class='chart-row'>"
    charts_html += img_block(chart_paths.get("refinement", ""), "Local refinement: objective before vs after coordinate polish")
    charts_html += "</div></div>"

    narrative_section = ""
    if narrative_html:
        narrative_section = (
            "<section class='narrative-wrap'><h2>Run narrative</h2>"
            f"{narrative_html}</section>"
        )

    gallery = "<section><h2>Physical simulation — best run</h2><p class='lead'>Heat fields from the full-resolution rerun at the optimal MW values.</p>"
    for ar in asset_rows:
        name = escape(ar["name"])
        gallery += f"<h3>{name}</h3><div class='grid'>"
        for label, rel in [
            ("City mask", ar.get("masks", "")),
            ("Final temperature", ar.get("final", "")),
            ("Anomaly", ar.get("anomaly", "")),
        ]:
            if not rel:
                continue
            url = pub_url(rel)
            gallery += (
                f"<figure class='thumb'><figcaption>{escape(label)}</figcaption>"
                f"<a href='{escape(url)}' target='_blank' rel='noopener'>"
                f"<img loading='lazy' src='{escape(url)}' alt='{escape(label)}'/></a></figure>"
            )
        if ar.get("gif"):
            gurl = pub_url(str(ar["gif"]))
            gallery += (
                f"<figure class='thumb'><figcaption>Heat plume GIF (wind arrow = advection direction)</figcaption>"
                f"<img src='{escape(gurl)}' alt='gif'/></figure>"
            )
        gallery += "</div>"
    gallery += "</section>"

    cap_lead = ""
    if has_caps:
        cap_lead = (
            "<p class='lead'>Sites with an <strong>Agent cap</strong> were limited after local review; "
            "reported optimal MW respects each cap (held within about 1 MW of the cap) while "
            "preserving the fleet total.</p>"
        )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Thermal load optimization — {escape(run_id)}</title>
<style>
:root {{ --bg:#0b1220; --card:#111827; --line:#1f2937; --txt:#e5e7eb; --muted:#94a3b8; --accent:#38bdf8; }}
* {{ box-sizing:border-box; }}
body {{ font-family: ui-sans-serif, system-ui, sans-serif; background:var(--bg); color:var(--txt); margin:0; padding:32px 24px 48px; line-height:1.55; }}
header {{ max-width:1100px; margin:0 auto 28px; padding:24px 28px; background:linear-gradient(135deg,#0f172a,#1e1b4b); border-radius:16px; border:1px solid var(--line); }}
h1 {{ margin:0 0 8px; font-size:1.65rem; letter-spacing:-0.02em; color:var(--accent); }}
.lead {{ color:var(--muted); font-size:0.95rem; margin:0 0 12px; max-width:900px; }}
h2 {{ color:var(--accent); font-size:1.15rem; margin:28px 0 12px; border-bottom:1px solid var(--line); padding-bottom:6px; }}
h3 {{ color:#a5f3fc; font-size:1rem; margin:18px 0 8px; }}
table {{ border-collapse:collapse; width:100%; margin:12px 0; font-size:13px; background:var(--card); border-radius:12px; overflow:hidden; }}
th,td {{ border:1px solid var(--line); padding:8px 10px; text-align:left; }}
th {{ background:#1e293b; color:#cbd5e1; font-weight:600; }}
tr:nth-child(even) td {{ background:rgba(15,23,42,0.45); }}
.charts figure.chart {{ margin:0 0 20px; background:var(--card); padding:12px; border-radius:12px; border:1px solid var(--line); max-width:100%; }}
.charts figure.chart img {{ width:100%; height:auto; border-radius:8px; display:block; }}
.chart-row {{ display:grid; grid-template-columns:1fr; gap:16px; max-width:1100px; }}
@media(min-width:900px){{ .chart-row {{ grid-template-columns:1fr 1fr; }} }}
figure.thumb {{ margin:0; background:var(--card); padding:10px; border-radius:12px; border:1px solid var(--line); max-width:420px; }}
figure.thumb img {{ max-width:100%; height:auto; border-radius:8px; display:block; }}
figcaption {{ font-size:12px; color:var(--muted); margin-bottom:8px; }}
.grid {{ display:flex; flex-wrap:wrap; gap:14px; }}
section {{ max-width:1100px; margin:0 auto; }}
.narrative-wrap {{ max-width:1100px; margin:0 auto 28px; padding:20px 24px; background:var(--card); border-radius:16px; border:1px solid var(--line); }}
.narrative-wrap h3 {{ color:#a5f3fc; margin:1.1em 0 0.4em; font-size:1.02rem; }}
.narrative-wrap p {{ margin:0.55em 0; color:var(--txt); }}
.narrative-wrap ul {{ margin:0.4em 0 0.6em 1.2em; color:#cbd5e1; }}
.narrative-wrap li {{ margin:0.25em 0; }}
.narrative-wrap code {{ font-size:0.88em; }}
</style></head><body>
<header>
<h1>Thermal load optimization report</h1>
<p class="lead">This run placed <strong>{extra_total_mw:.2f} MW</strong> of extra IT load across your sites. <strong>Lower objective is better.</strong> The scalar being minimized blends worst-site thermal rise (smooth max), imbalance of rises across sites, summed anomalies, hot-area footprint, and a mild worst absolute-temperature term (see <code>compute_total_objective</code> in the optimizer). Run id: <code>{escape(run_id)}</code></p>
<p class="lead">Best objective after full physics rerun: <strong>{best_objective:.4f}</strong></p>
</header>
<section>
{charts_html}
{narrative_section}
<h2>All candidates evaluated (GP search + EI)</h2>
<p class="lead">Every row is one simulated load split from Gaussian-process Bayesian optimization (expected improvement). The best seeds are refined locally.</p>
<table><thead>{g_head}</thead><tbody>{g_body}</tbody></table>
<h2>Local refinement</h2>
<table><thead>{rf_head}</thead><tbody>{rf_body}</tbody></table>
<h2>Best MW per site (after optimization)</h2>
{cap_lead}
<table><thead>{best_head}</thead><tbody>{best_body}</tbody></table>
</section>
{gallery}
<section style="margin-top:32px">
<h2>Machine-readable results</h2>
<p class="lead">Download <code>optimal_data.json</code> from the same folder for dashboards or further analysis.</p>
</section>
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
    *,
    min_loads_mw: Optional[np.ndarray] = None,
    max_loads_mw: Optional[np.ndarray] = None,
    optimizer_meta: Optional[Dict[str, Any]] = None,
    random_seed: int = RANDOM_SEED,
) -> Tuple[Path, Path, Dict[str, str]]:
    run_root.mkdir(parents=True, exist_ok=True)
    names = [str(dc.get("name", f"site_{i}")) for i, dc in enumerate(datacenters)]
    base_loads_mw = np.asarray(base_loads_mw, dtype=float).reshape(-1)
    n = len(datacenters)
    floor_min = (
        np.asarray(min_loads_mw, dtype=float).reshape(-1)
        if min_loads_mw is not None
        else np.maximum(base_loads_mw, MIN_LOAD_MW)
    )
    fleet_max = (
        np.asarray(max_loads_mw, dtype=float).reshape(-1)
        if max_loads_mw is not None
        else np.full(n, PER_SITE_FLEET_CAP_MW, dtype=float)
    )
    agent_ctx = (optimizer_meta or {}).get("final_objective_context")
    target_total = float(base_loads_mw.sum() + extra_total_load_mw)
    eff_min, eff_max = resolve_agent_load_bounds(floor_min, fleet_max, agent_ctx)
    best_loads_mw = project_loads_to_bounds(
        np.asarray(best_loads_mw, dtype=float).reshape(-1),
        eff_min,
        eff_max,
        target_total,
    )
    best_per = slim_eval_results(best_results)
    optimal_payload: Dict[str, Any] = {
        "run_id": run_id,
        "extra_total_load_mw": float(extra_total_load_mw),
        "base_loads_mw": base_loads_mw.tolist(),
        "best_loads_mw": best_loads_mw.tolist(),
        "best_objective": float(best_objective),
        "global_search": global_slim,
        "refinement": refined_slim,
        "best_per_site": best_per,
    }
    if optimizer_meta:
        optimal_payload.update(optimizer_meta)
    optimal_payload["effective_min_loads_mw"] = eff_min.tolist()
    optimal_payload["effective_max_loads_mw"] = eff_max.tolist()

    chart_paths = write_optimization_charts(
        run_root=run_root,
        global_slim=global_slim,
        refined_slim=refined_slim,
        datacenter_names=names,
        base_loads_mw=base_loads_mw.tolist(),
        best_loads_mw=best_loads_mw.tolist(),
        random_seed=random_seed,
        best_per_site=best_per,
    )
    optimal_payload["charts"] = chart_paths

    try:
        from optimization_report_narrator import generate_optimization_narrative_html

        narrative_html = generate_optimization_narrative_html(
            datacenters,
            optimal_payload,
            run_root=run_root,
        )
    except Exception as exc:
        narrative_html = (
            "<p><em>Narrative module unavailable: "
            f"{escape(str(exc))}</em></p>"
        )

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
        chart_paths=chart_paths,
        narrative_html=narrative_html,
        effective_min_loads_mw=eff_min.tolist(),
        effective_max_loads_mw=eff_max.tolist(),
    )
    html_path = run_root / "report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return html_path, json_path, chart_paths


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
    objective_context: Optional[Dict[str, Any]] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Yields progress/log dicts, then a final ``{"type": "complete", "result": ...}``.

    All loads are in **MW** (IT load into the physics wrapper).

    Search strategy:
        1. **Initial design**: latent vectors ``z ∈ R^{n-1}`` (softmax → extra-MW split),
           including an equal-split anchor and uniform random points.
        2. **Bayesian optimization**: ``sklearn.gaussian_process.GaussianProcessRegressor``
           (Matern kernel) on ``(z, objective)``, then **expected improvement** maximized
           by random search over ``z`` (``BO_EI_RANDOM_CANDIDATES`` proposals per step).
        3. **Local polish**: existing coordinate-wise load moves on the top-``k`` GP points.

    ``n_global_candidates`` is the total number of expensive physics evaluations in stage
    (1)+(2) before refinement.
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

    min_loads, max_loads = resolve_agent_load_bounds(
        min_loads, max_loads, objective_context
    )

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

    d_latent = max(0, n - 1)
    n_budget = max(int(n_global_candidates), max(BO_INIT_MIN, 2))

    if d_latent == 0:
        n_init = 1
        n_bo = 0
        n_budget = 1
    else:
        n_init = min(max(BO_INIT_MIN, n + 1), max(1, n_budget // 2))
        n_init = min(n_init, max(1, n_budget - 1))
        n_bo = max(0, n_budget - n_init)

    n_total_eval = n_init + n_bo

    if extra_total_load_mw > 0:
        step_sizes = [extra_total_load_mw / 4.0, extra_total_load_mw / 10.0]
    else:
        step_sizes = [0.25, 0.1]

    refine_ops_est = top_k_refine * len(step_sizes) * max(1, n * (n - 1))
    total_steps_est = n_total_eval + refine_ops_est

    yield {
        "type": "plan",
        "run_id": run_id,
        "phase": "init",
        "message": (
            f"GP Bayesian optimization: {n} sites, {n_total_eval} physics evals "
            f"({n_init} initial + {n_bo} EI), refine top-{top_k_refine}."
        ),
        "candidate_count": n_total_eval,
        "target_total_mw": target_total_load,
        "steps_total_estimate": int(total_steps_est),
    }

    global_history: List[Dict[str, Any]] = []
    X_rows: List[np.ndarray] = []
    y_list: List[float] = []

    init_z: List[np.ndarray] = []
    if d_latent == 0:
        init_z.append(np.zeros(0, dtype=float))
    else:
        init_z.append(np.zeros(d_latent, dtype=float))
        while len(init_z) < n_init:
            init_z.append(
                rng.uniform(-BO_Z_HALF_WIDTH, BO_Z_HALF_WIDTH, size=d_latent).astype(float)
            )
        init_z = init_z[:n_init]

    if verbose:
        print(
            f"\nGP BAYESIAN OPT — {n_total_eval} evals (init {n_init}, EI {n_bo}), "
            f"target {target_total_load:.3f} MW total, latent dim {d_latent}"
        )

    for idx in range(n_total_eval):
        steps_left = int(total_steps_est - idx)
        yield {
            "type": "progress",
            "phase": "global",
            "index": idx + 1,
            "total": n_total_eval,
            "steps_remaining": max(0, steps_left),
            "message": f"Evaluating BO point {idx + 1}/{n_total_eval}…",
        }

        if idx < n_init:
            z = init_z[idx].copy()
        elif d_latent == 0:
            z = np.zeros(0, dtype=float)
        else:
            gp = _fit_gp(np.vstack(X_rows), np.array(y_list, dtype=float), random_seed)
            if gp is not None:
                z = _suggest_next_z_ei(gp, np.array(y_list, dtype=float), rng, d_latent)
            else:
                z = rng.uniform(-BO_Z_HALF_WIDTH, BO_Z_HALF_WIDTH, size=d_latent)

        loads = decode_latent_z_to_loads_mw(
            z,
            base_loads=base_loads,
            min_loads=min_loads,
            max_loads=max_loads,
            extra_total_load_mw=extra_total_load_mw,
            target_total_load=target_total_load,
        )

        obj, results = evaluate_load_distribution(
            datacenters=datacenters,
            loads_mw=loads,
            cache=cache,
            run_id=run_id,
            save_gif=False,
            verbose=False,
            fast_evaluation=True,
            objective_context=objective_context,
        )

        if d_latent > 0:
            X_rows.append(z.astype(float).copy())
        y_list.append(float(obj))

        item: Dict[str, Any] = {
            "candidate_index": idx,
            "loads_mw": loads.copy(),
            "objective": float(obj),
            "results": results,
            "latent_z": z.astype(float).copy(),
        }
        global_history.append(item)

        phase = "init" if idx < n_init else "ei"
        msg = (
            f"BO [{phase}] {idx + 1}/{n_total_eval}: objective={obj:.4f} · loads MW="
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
        "message": (
            f"GP search done. Best obj so far: {global_history[0]['objective']:.4f}. "
            f"Starting local coordinate refinement…"
        ),
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
            "message": f"Refining from GP rank {rank + 1}/{len(top_candidates)} (obj={item['objective']:.4f})…",
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
            objective_context=objective_context,
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

    snapped = project_loads_to_bounds(
        np.asarray(best["loads_mw"], dtype=float),
        min_loads,
        max_loads,
        target_total_load,
    )
    if float(np.max(np.abs(snapped - np.asarray(best["loads_mw"], dtype=float)))) > 0.01:
        snap_obj, snap_results = evaluate_load_distribution(
            datacenters=datacenters,
            loads_mw=snapped,
            cache=cache,
            run_id=run_id,
            save_gif=False,
            verbose=False,
            fast_evaluation=True,
            objective_context=objective_context,
        )
        best["loads_mw"] = snapped
        best["objective"] = snap_obj
        best["results"] = snap_results

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
        "optimization_method": "sklearn_gaussian_process_expected_improvement",
        "bo_init_evals": int(n_init),
        "bo_ei_evals": int(n_bo),
        "bo_latent_dim": int(d_latent),
        "random_seed": int(random_seed),
        "objective_context": objective_context,
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
            save_numpy=False,
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