"""
Run one stable urban heat-transfer physical simulation around a central building.

Main behavior:
    - n_meters = 800.0
    - simulation steps = 3000
    - only the central building receives heat
    - the whole central building heats uniformly
    - heat dissipates from the real central-building shape
    - no whole-field solar/background heating
    - no checkerboard instability
    - no periodic np.roll diffusion artifacts

Weather/load logic:
    - Same IT load in cold and warm air.
    - Colder air dissipates heat better.
    - Warmer air dissipates heat worse.
    - Warmer air slightly increases cooling overhead.
    - Therefore, with the same load:
        16C ambient -> smaller final delta above ambient
        26C ambient -> larger final delta above ambient

Required:

    from buildings_latlon import latlon_to_central_building_mask

Expected returned dictionary:

    result["mask"]              -> full city buildings mask
    result["central_building"]  -> central building mask / heat source

Run:

    python run_one_physical_simulation.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from buildings_latlon import latlon_to_central_building_mask


# ============================================================
# 1. DEFAULT SIMULATION PARAMETERS
# ============================================================

DEFAULT_NX = 160
DEFAULT_NY = 160

DEFAULT_N_METERS = 400.0
DEFAULT_DX = DEFAULT_N_METERS / DEFAULT_NX
DEFAULT_DY = DEFAULT_N_METERS / DEFAULT_NY

DEFAULT_DT = 0.2
DEFAULT_STEPS = 3000
DEFAULT_SAVE_EVERY = 25

# Change this to compare 16C vs 26C.
DEFAULT_T_AIR = 6.0

DEFAULT_HUMIDITY = 0.60
DEFAULT_SOLAR_RADIATION = 650.0
DEFAULT_LOAD = 1.0+2

DEFAULT_WIND_X = 0.35
DEFAULT_WIND_Y = 0.16

RHO_AIR_20C = 1.204
CP_AIR = 1005.0
MIXING_HEIGHT = 18.0

ALPHA_OPEN = 4.0
ALPHA_BUILDING = 1.0
ALPHA_SOURCE_BUILDING = 1.6
ALPHA_ROAD_BOOST = 1.15

BASE_COOLING_RATE = 1.8e-4

BUILDING_HEAT_CAPACITY_FACTOR = 2.5
SOURCE_BUILDING_HEAT_CAPACITY_FACTOR = 2.2

BUILDING_WIND_REDUCTION = 0.35
SOURCE_BUILDING_WIND_REDUCTION = 0.05

# Solar is applied only to the central building.
CENTRAL_BUILDING_SOLAR_ABSORPTION = 0.55
CENTRAL_SOLAR_TO_HEAT_FRACTION = 0.012

LOAD_GAMMA = 1.65

# Moderate source multiplier for visible plume.
SOURCE_HEAT_MULTIPLIER = 2.2

MAX_ALLOWED_TEMPERATURE_C = 80.0

# Radiation constants.
STEFAN_BOLTZMANN = 5.670374419e-8
URBAN_EFFECTIVE_EMISSIVITY = 0.88

# Convective cooling coefficients.
BASE_CONVECTIVE_H_W_M2K = 4.0
WIND_CONVECTIVE_H_FACTOR = 3.0


# ============================================================
# 2. DATACENTER SPECS
# ============================================================

DATACENTER_SPECS: Dict[str, Any] = {
    "colocation_area_ft2": 108_976.0,
    "colocation_area_m2": 10_124.0,
    "num_floors": 4,
    "footprint_area_m2": 10_124.0 / 4.0,

    "building_type": "4-story concrete structure with concrete floor",
    "floor_type": "slab",

    "floor_load_capacity_psf": 200.0,
    "floor_load_capacity_kN_m2": 9.58,
    "floor_load_capacity_kg_m2": 9.58 * 1000.0 / 9.81,

    "cabinet_density_kVA": 5.0,
    "power_distribution": "120/208V",
    "utility_feeders": 2,

    "ups_configuration": "block redundant",
    "ups_redundancy": "N+1",

    "num_generators": 5,
    "generator_capacity_kW_each": 2500.0,
    "standby_power_redundancy": "N+1",

    "usable_generator_count": 4,
    "usable_standby_power_kW": 4 * 2500.0,
    "installed_standby_power_kW": 5 * 2500.0,

    "cooling_configuration": "centrifugal chillers and air handling units",
    "cooling_redundancy": "N+1",

    "assumed_window_to_wall_ratio": 0.08,
    "floor_height_m": 4.0,
    "wall_thickness_m": 0.25,
    "glass_thickness_m": 0.012,

    "concrete_Cv_J_m3K": 2.1e6,
    "glass_Cv_J_m3K": 1.9e6,

    "k_concrete_W_mK": 1.4,
    "k_glass_W_mK": 1.0,

    "U_wall_W_m2K": 0.55,
    "U_window_W_m2K": 2.6,
}


# ============================================================
# 3. DATACENTER THERMAL HELPERS
# ============================================================

def create_synthetic_datacenter_floor_plan(
    target_area_m2: float,
    dx_plan: float = 2.0,
    aspect_ratio: float = 1.45,
) -> Tuple[np.ndarray, float]:
    width_m = np.sqrt(target_area_m2 * aspect_ratio)
    height_m = target_area_m2 / width_m

    nx = int(np.ceil(width_m / dx_plan))
    ny = int(np.ceil(height_m / dx_plan))

    floor_plan = np.ones((nx, ny), dtype=bool)
    return floor_plan, dx_plan


def extract_floor_plan_geometry(
    floor_plan: np.ndarray,
    dx_plan: float,
) -> Dict[str, float]:
    floor_plan = floor_plan.astype(bool)

    area_m2 = float(np.sum(floor_plan) * dx_plan**2)

    padded = np.pad(floor_plan, pad_width=1, mode="constant", constant_values=False)

    center = padded[1:-1, 1:-1]
    north = padded[:-2, 1:-1]
    south = padded[2:, 1:-1]
    west = padded[1:-1, :-2]
    east = padded[1:-1, 2:]

    exposed_edges = (
        (center & ~north).sum()
        + (center & ~south).sum()
        + (center & ~west).sum()
        + (center & ~east).sum()
    )

    perimeter_m = float(exposed_edges * dx_plan)

    return {
        "footprint_area_m2": area_m2,
        "perimeter_m": perimeter_m,
    }


def extract_window_wall_properties_from_floor_plan(
    floor_plan: np.ndarray,
    dx_plan: float,
    num_floors: int,
    floor_height_m: float,
    window_to_wall_ratio: float,
) -> Dict[str, float]:
    geom = extract_floor_plan_geometry(floor_plan, dx_plan)

    total_height_m = num_floors * floor_height_m
    total_wall_area_m2 = geom["perimeter_m"] * total_height_m

    glass_area_m2 = total_wall_area_m2 * window_to_wall_ratio
    opaque_wall_area_m2 = total_wall_area_m2 - glass_area_m2

    return {
        "footprint_area_m2": geom["footprint_area_m2"],
        "perimeter_m": geom["perimeter_m"],
        "total_height_m": total_height_m,
        "total_wall_area_m2": total_wall_area_m2,
        "glass_area_m2": glass_area_m2,
        "opaque_wall_area_m2": opaque_wall_area_m2,
        "window_to_wall_ratio": window_to_wall_ratio,
    }


def compute_datacenter_effective_properties(
    specs: Dict[str, Any],
) -> Dict[str, Any]:
    colocation_area_m2 = float(specs["colocation_area_m2"])
    footprint_area_m2 = float(specs["footprint_area_m2"])

    floor_plan, dx_plan = create_synthetic_datacenter_floor_plan(
        target_area_m2=footprint_area_m2,
        dx_plan=2.0,
        aspect_ratio=1.45,
    )

    envelope = extract_window_wall_properties_from_floor_plan(
        floor_plan=floor_plan,
        dx_plan=dx_plan,
        num_floors=int(specs["num_floors"]),
        floor_height_m=float(specs["floor_height_m"]),
        window_to_wall_ratio=float(specs["assumed_window_to_wall_ratio"]),
    )

    installed_standby_power_W = float(specs["installed_standby_power_kW"]) * 1000.0
    usable_standby_power_W = float(specs["usable_standby_power_kW"]) * 1000.0

    installed_power_density_W_m2 = installed_standby_power_W / colocation_area_m2
    usable_power_density_W_m2 = usable_standby_power_W / colocation_area_m2

    concrete_Cv = float(specs["concrete_Cv_J_m3K"])
    glass_Cv = float(specs["glass_Cv_J_m3K"])

    wall_thickness = float(specs["wall_thickness_m"])
    glass_thickness = float(specs["glass_thickness_m"])

    U_wall = float(specs["U_wall_W_m2K"])
    U_window = float(specs["U_window_W_m2K"])

    opaque_wall_area_m2 = envelope["opaque_wall_area_m2"]
    glass_area_m2 = envelope["glass_area_m2"]

    C_wall_total_J_K = concrete_Cv * opaque_wall_area_m2 * wall_thickness
    C_glass_total_J_K = glass_Cv * glass_area_m2 * glass_thickness

    C_envelope_total_J_K = C_wall_total_J_K + C_glass_total_J_K
    C_envelope_areal_J_m2K = C_envelope_total_J_K / footprint_area_m2

    effective_slab_depth = 0.20
    C_slab_areal_J_m2K = concrete_Cv * effective_slab_depth

    C_envelope = C_envelope_areal_J_m2K + C_slab_areal_J_m2K

    UA_wall_W_K = U_wall * opaque_wall_area_m2
    UA_window_W_K = U_window * glass_area_m2
    UA_total_W_K = UA_wall_W_K + UA_window_W_K

    H_envelope_areal_W_m2K = UA_total_W_K / footprint_area_m2

    assumed_pue = 1.35
    heat_rejection_fraction = 0.90

    return {
        "floor_plan": floor_plan,
        "floor_plan_dx_m": dx_plan,

        "colocation_area_m2": colocation_area_m2,
        "footprint_area_m2": footprint_area_m2,

        "installed_standby_power_W": installed_standby_power_W,
        "usable_standby_power_W": usable_standby_power_W,

        "installed_power_density_W_m2": installed_power_density_W_m2,
        "usable_power_density_W_m2": usable_power_density_W_m2,

        "perimeter_m": envelope["perimeter_m"],
        "total_height_m": envelope["total_height_m"],
        "total_wall_area_m2": envelope["total_wall_area_m2"],
        "opaque_wall_area_m2": envelope["opaque_wall_area_m2"],
        "glass_area_m2": envelope["glass_area_m2"],
        "window_to_wall_ratio": envelope["window_to_wall_ratio"],

        "C_wall_total_J_K": C_wall_total_J_K,
        "C_glass_total_J_K": C_glass_total_J_K,
        "C_envelope_total_J_K": C_envelope_total_J_K,
        "C_envelope": C_envelope,

        "UA_wall_W_K": UA_wall_W_K,
        "UA_window_W_K": UA_window_W_K,
        "UA_total_W_K": UA_total_W_K,
        "H_envelope_areal_W_m2K": H_envelope_areal_W_m2K,

        "pue": assumed_pue,
        "heat_rejection_fraction": heat_rejection_fraction,
    }


# ============================================================
# 4. WEATHER / AMBIENT PHYSICS
# ============================================================

def air_density_from_temperature_C(T_air_C: float) -> float:
    """
    Colder air is denser, so it can absorb/remove heat slightly better.
    """
    T_ref_K = 293.15
    T_air_K = T_air_C + 273.15
    return RHO_AIR_20C * T_ref_K / T_air_K


def air_areal_heat_capacity(T_air_C: float) -> float:
    """
    Effective heat capacity of the urban air column.
    """
    rho_air = air_density_from_temperature_C(T_air_C)
    return rho_air * CP_AIR * MIXING_HEIGHT


def weather_dissipation_factor(T_air_C: float, humidity: float) -> float:
    """
    Main weather logic.

    Colder air dissipates heat better.
    Warmer air dissipates heat worse.

    Example with humidity = 0.60:
        T_air = 16C -> factor > 1
        T_air = 26C -> factor < 1

    Humidity slightly reduces dissipation.
    """
    temp_factor = 1.0 + 0.055 * (20.0 - T_air_C)
    humidity_factor = 1.0 - 0.20 * humidity

    factor = temp_factor * humidity_factor

    return float(np.clip(factor, 0.45, 1.80))


def datacenter_cooling_overhead_factor(T_air_C: float) -> float:
    """
    Same useful IT load, but warmer outdoor air makes cooling less efficient.

    This adds some extra heat rejection overhead in warm weather.
    """
    factor = 1.0 + 0.025 * (T_air_C - 20.0)
    return float(np.clip(factor, 0.80, 1.35))


def convective_heat_loss_W_m2(
    T: np.ndarray,
    T_air_C: float,
    wind_speed: np.ndarray,
    humidity: float,
) -> np.ndarray:
    """
    Convective heat loss:

        q_conv = h_weather * (T - T_air)

    Cold weather increases h_weather, so the same heat load produces
    a smaller temperature rise.
    """
    weather_factor = weather_dissipation_factor(T_air_C, humidity)

    h = (
        BASE_CONVECTIVE_H_W_M2K
        + WIND_CONVECTIVE_H_FACTOR * np.sqrt(np.maximum(wind_speed, 0.0))
    )

    h = h * weather_factor

    return h * (T - T_air_C)


def radiative_heat_loss_W_m2(
    T: np.ndarray,
    T_air_C: float,
) -> np.ndarray:
    """
    Longwave radiation:

        q_rad = epsilon * sigma * (T_K^4 - T_air_K^4)
    """
    T_K = np.maximum(T + 273.15, 1.0)
    T_air_K = T_air_C + 273.15

    return URBAN_EFFECTIVE_EMISSIVITY * STEFAN_BOLTZMANN * (T_K**4 - T_air_K**4)


# ============================================================
# 5. MASK UTILITIES
# ============================================================

def resize_nearest(
    mask: np.ndarray,
    target_shape: Tuple[int, int],
) -> np.ndarray:
    mask = np.asarray(mask)
    nx, ny = target_shape

    ix = np.linspace(0, mask.shape[0] - 1, nx).round().astype(int)
    iy = np.linspace(0, mask.shape[1] - 1, ny).round().astype(int)

    return mask[np.ix_(ix, iy)]


def extract_masks_from_latlon_result(
    result: Any,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    metadata: Dict[str, Any] = {}

    if isinstance(result, dict):
        metadata = dict(result)

        if "mask" not in result:
            raise KeyError(
                "Could not find full buildings mask. Expected key 'mask'. "
                f"Returned keys were: {', '.join(result.keys())}"
            )

        if "central_building" not in result:
            raise KeyError(
                "Could not find central building mask. Expected key 'central_building'. "
                f"Returned keys were: {', '.join(result.keys())}"
            )

        buildings = np.asarray(result["mask"])
        central = np.asarray(result["central_building"])

        return buildings, central, metadata

    if isinstance(result, (tuple, list)):
        if len(result) < 2:
            raise ValueError(
                "latlon_to_central_building_mask returned fewer than 2 values."
            )

        buildings = np.asarray(result[0])
        central = np.asarray(result[1])

        if len(result) >= 3 and isinstance(result[2], dict):
            metadata = dict(result[2])

        return buildings, central, metadata

    raise TypeError(
        "Unsupported return type from latlon_to_central_building_mask: "
        f"{type(result).__name__}."
    )


def clean_city_masks(
    buildings: np.ndarray,
    central_building_mask: np.ndarray,
    Nx: int,
    Ny: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if buildings.ndim != 2 or central_building_mask.ndim != 2:
        raise ValueError(
            "Both returned masks must be 2D arrays. Got "
            f"buildings shape {buildings.shape}, "
            f"central shape {central_building_mask.shape}."
        )

    if buildings.shape != (Nx, Ny):
        buildings = resize_nearest(buildings, (Nx, Ny))

    if central_building_mask.shape != (Nx, Ny):
        central_building_mask = resize_nearest(central_building_mask, (Nx, Ny))

    buildings = buildings.astype(bool)
    central_building_mask = central_building_mask.astype(bool)

    buildings = buildings | central_building_mask

    if central_building_mask.sum() == 0:
        raise ValueError(
            "central_building_mask is empty. Check lat/lon or use_nearest_if_not_inside."
        )

    return buildings, central_building_mask


# ============================================================
# 6. STABLE NUMERICAL OPERATORS
# ============================================================

def neighbor_arrays_edge(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Neighbor arrays with edge padding.

    This avoids periodic wrap-around artifacts from np.roll.
    """
    padded = np.pad(T, pad_width=1, mode="edge")

    T_ip = padded[2:, 1:-1]
    T_im = padded[:-2, 1:-1]
    T_jp = padded[1:-1, 2:]
    T_jm = padded[1:-1, :-2]

    return T_ip, T_im, T_jp, T_jm


def upwind_gradient(
    T: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    dx: float,
    dy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    T_ip, T_im, T_jp, T_jm = neighbor_arrays_edge(T)

    dTdx_back = (T - T_im) / dx
    dTdx_forw = (T_ip - T) / dx

    dTdy_back = (T - T_jm) / dy
    dTdy_forw = (T_jp - T) / dy

    dTdx = np.where(u >= 0.0, dTdx_back, dTdx_forw)
    dTdy = np.where(v >= 0.0, dTdy_back, dTdy_forw)

    return dTdx, dTdy


def variable_diffusion(
    T: np.ndarray,
    alpha: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """
    Stable non-periodic variable diffusion:

        div(alpha grad T)
    """
    T_ip, T_im, T_jp, T_jm = neighbor_arrays_edge(T)

    alpha_padded = np.pad(alpha, pad_width=1, mode="edge")

    alpha_c = alpha_padded[1:-1, 1:-1]
    alpha_ip = 0.5 * (alpha_c + alpha_padded[2:, 1:-1])
    alpha_im = 0.5 * (alpha_c + alpha_padded[:-2, 1:-1])
    alpha_jp = 0.5 * (alpha_c + alpha_padded[1:-1, 2:])
    alpha_jm = 0.5 * (alpha_c + alpha_padded[1:-1, :-2])

    diff_x = (
        alpha_ip * (T_ip - T)
        - alpha_im * (T - T_im)
    ) / dx**2

    diff_y = (
        alpha_jp * (T_jp - T)
        - alpha_jm * (T - T_jm)
    ) / dy**2

    return diff_x + diff_y


def apply_open_boundary(
    T: np.ndarray,
    T_air: float,
) -> np.ndarray:
    """
    Weak open boundary relaxation to ambient.
    """
    T[0, :] = 0.6 * T[1, :] + 0.4 * T_air
    T[-1, :] = 0.6 * T[-2, :] + 0.4 * T_air
    T[:, 0] = 0.6 * T[:, 1] + 0.4 * T_air
    T[:, -1] = 0.6 * T[:, -2] + 0.4 * T_air

    return T


def enforce_uniform_central_building_temperature(
    T: np.ndarray,
    central_building_mask: np.ndarray,
) -> np.ndarray:
    """
    Treat the central building as one thermal body.
    """
    if central_building_mask.sum() == 0:
        return T

    central_temp = float(T[central_building_mask].mean())
    T[central_building_mask] = central_temp

    return T


def local_smooth_anomaly(
    T: np.ndarray,
    T_air: float,
    central_building_mask: np.ndarray,
    strength: float = 0.035,
) -> np.ndarray:
    """
    Mild numerical anti-checker smoothing on temperature anomaly.

    It does not create heat. It only damps cell-scale oscillations.
    """
    anomaly = T - T_air

    A_ip, A_im, A_jp, A_jm = neighbor_arrays_edge(anomaly)
    neighbor_mean = 0.25 * (A_ip + A_im + A_jp + A_jm)

    anomaly = (1.0 - strength) * anomaly + strength * neighbor_mean

    T_new = T_air + anomaly
    T_new = enforce_uniform_central_building_temperature(T_new, central_building_mask)

    return T_new


# ============================================================
# 7. MAIN PHYSICAL MODEL
# ============================================================

def simulate_city_heat_one_case(
    buildings: np.ndarray,
    central_building_mask: np.ndarray,
    load: float,
    wind_x: float,
    wind_y: float,
    T_air: float = DEFAULT_T_AIR,
    humidity: float = DEFAULT_HUMIDITY,
    solar_radiation: float = DEFAULT_SOLAR_RADIATION,
    datacenter_specs: Dict[str, Any] | None = None,
    dx: float = DEFAULT_DX,
    dy: float = DEFAULT_DY,
    dt: float = DEFAULT_DT,
    steps: int = DEFAULT_STEPS,
    save_every: int = DEFAULT_SAVE_EVERY,
    return_states: bool = True,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:

    if datacenter_specs is None:
        datacenter_specs = DATACENTER_SPECS

    buildings = buildings.astype(bool)
    central_building_mask = central_building_mask.astype(bool)

    buildings = buildings | central_building_mask
    central_building_mask = central_building_mask & buildings

    if central_building_mask.sum() == 0:
        raise ValueError("central_building_mask is empty.")

    nx, ny = buildings.shape

    dc_props = compute_datacenter_effective_properties(datacenter_specs)

    # Initial condition: whole city starts at ambient air temperature.
    T = np.ones((nx, ny), dtype=float) * T_air

    source_mask = central_building_mask.astype(float)
    source_cell_count = max(int(central_building_mask.sum()), 1)

    C_air = air_areal_heat_capacity(T_air)

    weather_factor = weather_dissipation_factor(T_air, humidity)
    cooling_overhead_factor = datacenter_cooling_overhead_factor(T_air)

    street_mask = ~buildings

    alpha = np.where(buildings, ALPHA_BUILDING, ALPHA_OPEN)
    alpha = np.where(street_mask, alpha * ALPHA_ROAD_BOOST, alpha)
    alpha = np.where(central_building_mask, ALPHA_SOURCE_BUILDING, alpha)

    # Cold weather increases effective turbulent dissipation.
    # Warm weather reduces effective turbulent dissipation.
    alpha = alpha * weather_factor

    alpha_max = float(alpha.max())
    stable_dt_limit = 0.24 * min(dx, dy) ** 2 / max(alpha_max, 1e-12)

    if dt > stable_dt_limit:
        raise ValueError(
            f"Unstable explicit diffusion setting: dt={dt:.4f}, "
            f"stable limit is about {stable_dt_limit:.4f}. "
            "Reduce DEFAULT_DT or ALPHA_OPEN."
        )

    wind_reduction = np.where(buildings, BUILDING_WIND_REDUCTION, 1.0)
    wind_reduction = np.where(
        central_building_mask,
        SOURCE_BUILDING_WIND_REDUCTION,
        wind_reduction,
    )

    u = wind_x * wind_reduction
    v = wind_y * wind_reduction
    wind_speed = np.sqrt(u**2 + v**2)

    C_eff = np.where(
        buildings,
        C_air * BUILDING_HEAT_CAPACITY_FACTOR,
        C_air,
    )

    C_eff = np.where(
        central_building_mask,
        C_air * SOURCE_BUILDING_HEAT_CAPACITY_FACTOR + dc_props["C_envelope"],
        C_eff,
    )

    # Cold weather has larger weather_factor -> stronger cooling.
    # Warm weather has smaller weather_factor -> weaker cooling.
    cooling_rate = BASE_COOLING_RATE * weather_factor

    nonlinear_load = max(load, 0.0) ** LOAD_GAMMA

    # Same useful IT load.
    Q_power_density = dc_props["usable_power_density_W_m2"] * nonlinear_load

    # Warmer weather makes cooling less efficient and increases rejected heat overhead.
    Q_facility_density = (
        Q_power_density
        * dc_props["pue"]
        * cooling_overhead_factor
    )

    Q_waste_density = Q_facility_density * dc_props["heat_rejection_fraction"]

    Q_central_solar = (
        solar_radiation
        * CENTRAL_BUILDING_SOLAR_ABSORPTION
        * CENTRAL_SOLAR_TO_HEAT_FRACTION
    )

    # Heat source exists only on the central building.
    Q_source = (
        SOURCE_HEAT_MULTIPLIER
        * (Q_waste_density + Q_central_solar)
        * source_mask
    )

    H_env = dc_props["H_envelope_areal_W_m2K"]

    states = []

    T = enforce_uniform_central_building_temperature(T, central_building_mask)

    for n in range(steps):
        dTdx, dTdy = upwind_gradient(T, u, v, dx, dy)
        dTdt_adv = -(u * dTdx + v * dTdy)

        dTdt_diff = variable_diffusion(T, alpha, dx, dy)

        # Only central building is heated.
        dTdt_source = Q_source / C_eff

        # Bulk weather cooling.
        dTdt_bulk_cool = -cooling_rate * (T - T_air)

        # Convective cooling in W/m2 converted to K/s.
        q_conv = convective_heat_loss_W_m2(
            T=T,
            T_air_C=T_air,
            wind_speed=wind_speed,
            humidity=humidity,
        )
        dTdt_conv = -q_conv / C_eff

        # Radiation cooling in W/m2 converted to K/s.
        q_rad = radiative_heat_loss_W_m2(
            T=T,
            T_air_C=T_air,
        )
        dTdt_rad = -q_rad / C_eff

        # Wall/window loss only from central building.
        dTdt_envelope_loss = -source_mask * H_env * (T - T_air) / C_eff

        # Gentle nonlinear stabilization for very hot cells.
        excess = np.maximum(T - T_air, 0.0)
        dTdt_hot_cooling = -1.1e-6 * excess**2

        dTdt = (
            dTdt_adv
            + dTdt_diff
            + dTdt_source
            + dTdt_bulk_cool
            + dTdt_conv
            + dTdt_rad
            + dTdt_envelope_loss
            + dTdt_hot_cooling
        )

        T = T + dt * dTdt

        # Central building is one thermal body.
        T = enforce_uniform_central_building_temperature(T, central_building_mask)

        # Mild anti-checker smoothing.
        if n % 4 == 0:
            T = local_smooth_anomaly(
                T=T,
                T_air=T_air,
                central_building_mask=central_building_mask,
                strength=0.035,
            )

        T = apply_open_boundary(T, T_air)

        T = np.clip(T, T_air - 2.0, MAX_ALLOWED_TEMPERATURE_C)

        T = enforce_uniform_central_building_temperature(T, central_building_mask)

        if return_states and n % save_every == 0:
            states.append(T.copy())

    anomaly = T - T_air

    central_temp = float(T[central_building_mask].mean())
    central_anomaly = central_temp - T_air

    thermal_risk_objective = float(
        1.0 * float(T.max())
        + 0.01 * int(np.sum(T > 35.0))
        + 0.03 * int(np.sum(T > 40.0))
        + 0.10 * int(np.sum(T > 45.0))
    )

    metrics = {
        "load": float(load),
        "source_heat_multiplier": float(SOURCE_HEAT_MULTIPLIER),
        "wind_x_m_s": float(wind_x),
        "wind_y_m_s": float(wind_y),
        "T_air_C": float(T_air),
        "humidity": float(humidity),
        "solar_radiation_W_m2": float(solar_radiation),

        "weather_dissipation_factor": float(weather_factor),
        "datacenter_cooling_overhead_factor": float(cooling_overhead_factor),
        "air_density_kg_m3": float(air_density_from_temperature_C(T_air)),
        "air_areal_heat_capacity_J_m2K": float(C_air),

        "dt_s": float(dt),
        "steps": int(steps),
        "simulation_time_s": float(dt * steps),
        "simulation_time_min": float(dt * steps / 60.0),

        "dx_m": float(dx),
        "dy_m": float(dy),
        "alpha_max_m2_s": float(alpha_max),
        "stable_dt_limit_s": float(stable_dt_limit),

        "central_building_temperature_C": central_temp,
        "central_building_anomaly_C": float(central_anomaly),

        "max_temp_C": float(T.max()),
        "mean_temp_C": float(T.mean()),
        "max_anomaly_C": float(anomaly.max()),
        "mean_anomaly_C": float(anomaly.mean()),
        "total_anomaly_C_cells": float(anomaly.sum()),

        "hot_area_gt_0p2C_cells": int(np.sum(anomaly > 0.2)),
        "hot_area_gt_0p5C_cells": int(np.sum(anomaly > 0.5)),
        "hot_area_gt_1C_cells": int(np.sum(anomaly > 1.0)),
        "hot_area_gt_2C_cells": int(np.sum(anomaly > 2.0)),
        "hot_area_gt_5C_cells": int(np.sum(anomaly > 5.0)),

        "hot_area_gt_30C_cells": int(np.sum(T > 30.0)),
        "hot_area_gt_35C_cells": int(np.sum(T > 35.0)),
        "hot_area_gt_40C_cells": int(np.sum(T > 40.0)),
        "hot_area_gt_45C_cells": int(np.sum(T > 45.0)),
        "thermal_risk_objective": thermal_risk_objective,

        "building_cells": int(buildings.sum()),
        "central_building_cells": int(source_cell_count),

        "usable_power_density_W_m2": float(dc_props["usable_power_density_W_m2"]),
        "source_heat_density_W_m2": float(Q_waste_density),
        "effective_source_heat_density_W_m2": float(
            SOURCE_HEAT_MULTIPLIER * Q_waste_density
        ),
        "central_solar_heat_density_W_m2": float(Q_central_solar),

        "window_to_wall_ratio": float(dc_props["window_to_wall_ratio"]),
        "glass_area_m2": float(dc_props["glass_area_m2"]),
        "opaque_wall_area_m2": float(dc_props["opaque_wall_area_m2"]),
        "total_wall_area_m2": float(dc_props["total_wall_area_m2"]),
        "UA_total_W_K": float(dc_props["UA_total_W_K"]),
        "H_envelope_areal_W_m2K": float(dc_props["H_envelope_areal_W_m2K"]),
    }

    states_arr = np.array(states) if return_states else np.empty((0, nx, ny))

    return metrics, T, states_arr


# ============================================================
# 8. PLOTTING / SAVING
# ============================================================

def plot_masks(
    buildings: np.ndarray,
    central_building_mask: np.ndarray,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.imshow(buildings.T, origin="lower", cmap="gray_r")

    ax.contour(
        central_building_mask.T,
        levels=[0.5],
        colors="red",
        linewidths=1.5,
    )

    ax.set_title("City building mask with central heat-source building")
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")

    fig.tight_layout()
    fig.savefig(output_dir / "01_building_masks.png", dpi=200)
    plt.close(fig)


def plot_final_temperature(
    T_final: np.ndarray,
    buildings: np.ndarray,
    central_building_mask: np.ndarray,
    T_air: float,
    wind_x: float,
    wind_y: float,
    output_dir: Path,
) -> None:
    anomaly = T_final - T_air

    fig, ax = plt.subplots(figsize=(7, 6))

    vmax_temp = max(float(np.percentile(T_final, 99.7)), T_air + 1.0)

    im = ax.imshow(
        T_final.T,
        origin="lower",
        cmap="inferno",
        vmin=T_air,
        vmax=vmax_temp,
    )

    ax.contour(
        buildings.T,
        levels=[0.5],
        colors="cyan",
        linewidths=0.25,
    )

    ax.contour(
        central_building_mask.T,
        levels=[0.5],
        colors="white",
        linewidths=1.4,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Temperature [°C]")

    ax.set_title(f"Final temperature field, ambient = {T_air:.1f} °C")
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")

    arrow_scale = 22

    ax.arrow(
        8,
        8,
        arrow_scale * wind_x,
        arrow_scale * wind_y,
        color="white",
        width=0.6,
        head_width=4,
        length_includes_head=True,
    )

    ax.text(8, 17, "wind", color="white", fontsize=11)

    fig.tight_layout()
    fig.savefig(output_dir / "02_final_temperature.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))

    vmax_anomaly = max(float(np.percentile(anomaly, 99.7)), 0.5)

    im = ax.imshow(
        anomaly.T,
        origin="lower",
        cmap="coolwarm",
        vmin=0.0,
        vmax=vmax_anomaly,
    )

    ax.contour(
        buildings.T,
        levels=[0.5],
        colors="black",
        linewidths=0.25,
    )

    ax.contour(
        central_building_mask.T,
        levels=[0.5],
        colors="yellow",
        linewidths=1.4,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Temperature anomaly [°C]")

    ax.set_title(f"Final heat-island anomaly, ambient = {T_air:.1f} °C")
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")

    fig.tight_layout()
    fig.savefig(output_dir / "03_final_anomaly.png", dpi=200)
    plt.close(fig)


def save_animation(
    states: np.ndarray,
    buildings: np.ndarray,
    central_building_mask: np.ndarray,
    T_air: float,
    wind_x: float,
    wind_y: float,
    save_every: int,
    dt: float,
    output_dir: Path,
) -> None:
    if states.size == 0 or len(states) < 2:
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    vmax = max(float(np.percentile(states, 99.7)), T_air + 1.0)

    im = ax.imshow(
        states[0].T,
        origin="lower",
        cmap="inferno",
        vmin=T_air,
        vmax=vmax,
    )

    ax.contour(
        buildings.T,
        levels=[0.5],
        colors="cyan",
        linewidths=0.20,
    )

    ax.contour(
        central_building_mask.T,
        levels=[0.5],
        colors="white",
        linewidths=1.2,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Temperature [°C]")

    title = ax.set_title(f"Heat plume evolution, ambient = {T_air:.1f} °C")

    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")

    arrow_scale = 22

    ax.arrow(
        8,
        8,
        arrow_scale * wind_x,
        arrow_scale * wind_y,
        color="white",
        width=0.6,
        head_width=4,
        length_includes_head=True,
    )

    ax.text(8, 17, "wind", color="white", fontsize=11)

    def update(frame: int):
        im.set_data(states[frame].T)

        minutes = frame * save_every * dt / 60.0
        title.set_text(
            f"Heat plume evolution, ambient = {T_air:.1f} °C, t = {minutes:.1f} min"
        )

        return [im, title]

    ani = FuncAnimation(
        fig,
        update,
        frames=len(states),
        interval=60,
        blit=False,
    )

    try:
        ani.save(
            output_dir / "04_heat_plume_animation.gif",
            writer="pillow",
            fps=15,
        )
    except Exception as exc:
        print(f"Could not save GIF animation. Reason: {exc}")

    plt.close(fig)


def _humidity_fraction(humidity: float) -> float:
    h = float(humidity)
    if h > 1.0 + 1e-6:
        return float(np.clip(h / 100.0, 0.0, 1.0))
    return float(np.clip(h, 0.0, 1.0))


def mw_it_load_multiplier(load_mw: float, reference_mw: float = 68.0) -> float:
    """
    Map dashboard MW to the model's dimensionless IT load input.

    Reference: ~68 MW corresponds to internal load factor ~3 (similar to DEFAULT_LOAD).
    """
    ref = max(float(reference_mw), 1.0)
    x = (float(load_mw) / ref) * 3.0
    return float(np.clip(x, 0.3, 30.0))


def wind_cardinal_to_components(
    wind_speed_m_s: float,
    wind_direction: str,
    scale: float = 0.1,
) -> Tuple[float, float]:
    """
    Convert UI wind (speed m/s, cardinal FROM which wind blows) to (wind_x, wind_y)
    advection components for ``simulate_city_heat_one_case``.
    """
    text = (wind_direction or "N").strip().upper()
    cardinals: Dict[str, float] = {
        "N": 0.0,
        "NE": 45.0,
        "E": 90.0,
        "SE": 135.0,
        "S": 180.0,
        "SW": 225.0,
        "W": 270.0,
        "NW": 315.0,
    }
    deg = cardinals.get(text)
    if deg is None:
        for key, val in cardinals.items():
            if text.startswith(key):
                deg = val
                break
    if deg is None:
        deg = 0.0

    rad = math.radians(deg + 180.0)
    s = max(float(wind_speed_m_s), 0.0) * scale
    return s * math.sin(rad), s * math.cos(rad)


def run_physical_simulation_for_params(
    *,
    lat: float,
    lon: float,
    load_mw: float,
    temp_c: float,
    humidity: float,
    solar_wm2: float,
    wind_speed_m_s: float,
    wind_direction: str,
    output_dir: Path,
    wind_x: float | None = None,
    wind_y: float | None = None,
    reverse_geocode: bool = True,
    all_touched: bool = True,
    use_nearest_if_not_inside: bool = True,
    n_meters: float = DEFAULT_N_METERS,
    Nx: int = DEFAULT_NX,
    Ny: int = DEFAULT_NY,
    dt: float = DEFAULT_DT,
    steps: int = DEFAULT_STEPS,
    save_every: int = DEFAULT_SAVE_EVERY,
    save_gif: bool = True,
    return_states: bool | None = None,
    write_disk: bool = True,
    save_plots: bool = True,
    save_numpy: bool = True,
    save_metadata_json: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Full pipeline: fetch masks at (lat, lon), run heat PDE, write PNG/GIF under ``output_dir``.

    Returns a dict with ``metrics`` (float stats) and ``output_dir`` (Path).

    ``write_disk=False`` skips all file output (metrics-only in memory), for fast optimizers.
    ``return_states=None`` defaults to True when ``save_gif`` else False.
    """
    load = mw_it_load_multiplier(load_mw)
    T_air = float(temp_c)
    humidity_f = _humidity_fraction(humidity)
    solar_radiation = float(solar_wm2)

    if wind_x is not None and wind_y is not None:
        wx, wy = float(wind_x), float(wind_y)
    else:
        wx, wy = wind_cardinal_to_components(wind_speed_m_s, wind_direction)

    dx = n_meters / Nx
    dy = n_meters / Ny

    if verbose:
        print("Loading city masks from buildings_latlon.py...")

    result = latlon_to_central_building_mask(
        lat=lat,
        lon=lon,
        n_meters=n_meters,
        Nx=Nx,
        Ny=Ny,
        reverse_geocode=reverse_geocode,
        all_touched=all_touched,
        use_nearest_if_not_inside=use_nearest_if_not_inside,
    )

    buildings, central_building_mask, metadata = extract_masks_from_latlon_result(result)

    buildings, central_building_mask = clean_city_masks(
        buildings=buildings,
        central_building_mask=central_building_mask,
        Nx=Nx,
        Ny=Ny,
    )

    if verbose:
        print("Running one stable physical heat simulation...")

    if return_states is None:
        return_states = bool(save_gif)

    metrics, T_final, states = simulate_city_heat_one_case(
        buildings=buildings,
        central_building_mask=central_building_mask,
        load=load,
        wind_x=wx,
        wind_y=wy,
        T_air=T_air,
        humidity=humidity_f,
        solar_radiation=solar_radiation,
        datacenter_specs=DATACENTER_SPECS,
        dx=dx,
        dy=dy,
        dt=dt,
        steps=steps,
        save_every=save_every,
        return_states=return_states,
    )

    if write_disk:
        save_outputs(
            output_dir=output_dir,
            buildings=buildings,
            central_building_mask=central_building_mask,
            T_final=T_final,
            states=states,
            metrics=metrics,
            metadata=metadata,
            T_air=T_air,
            wind_x=wx,
            wind_y=wy,
            save_every=save_every,
            dt=dt,
            save_gif=save_gif,
            save_plots=save_plots,
            save_numpy=save_numpy,
            save_metadata_json=save_metadata_json,
        )

    return {"metrics": metrics, "output_dir": output_dir}


def save_outputs(
    output_dir: Path,
    buildings: np.ndarray,
    central_building_mask: np.ndarray,
    T_final: np.ndarray,
    states: np.ndarray,
    metrics: Dict[str, float],
    metadata: Dict[str, Any],
    T_air: float,
    wind_x: float,
    wind_y: float,
    save_every: int,
    dt: float,
    save_gif: bool,
    save_plots: bool = True,
    save_numpy: bool = True,
    save_metadata_json: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_numpy:
        np.save(output_dir / "buildings_mask.npy", buildings.astype(bool))
        np.save(output_dir / "central_building_mask.npy", central_building_mask.astype(bool))
        np.save(output_dir / "temperature_final_C.npy", T_final)

        if states.size > 0:
            np.save(output_dir / "temperature_states_C.npy", states)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if save_metadata_json:
        serializable_metadata: Dict[str, Any] = {}

        for key, value in metadata.items():
            if key in ["mask", "central_building"]:
                continue

            if isinstance(value, (str, int, float, bool)) or value is None:
                serializable_metadata[key] = value
            elif isinstance(value, (list, tuple)):
                serializable_metadata[key] = [str(x) for x in value]
            else:
                serializable_metadata[key] = str(value)

        with open(output_dir / "latlon_mask_metadata.json", "w", encoding="utf-8") as f:
            json.dump(serializable_metadata, f, indent=2)

    if save_plots:
        plot_masks(
            buildings=buildings,
            central_building_mask=central_building_mask,
            output_dir=output_dir,
        )

        plot_final_temperature(
            T_final=T_final,
            buildings=buildings,
            central_building_mask=central_building_mask,
            T_air=T_air,
            wind_x=wind_x,
            wind_y=wind_y,
            output_dir=output_dir,
        )

    if save_gif and states.size > 0:
        save_animation(
            states=states,
            buildings=buildings,
            central_building_mask=central_building_mask,
            T_air=T_air,
            wind_x=wind_x,
            wind_y=wind_y,
            save_every=save_every,
            dt=dt,
            output_dir=output_dir,
        )


# ============================================================
# 9. MAIN
# ============================================================

def main() -> None:
    """
    CLI entry: same defaults as before, routed through ``run_physical_simulation_for_params``.
    """
    output_dir = Path("simulation_outputs")
    save_gif = True
    show_final_plot = False

    # ~68 MW ≈ internal load factor 3 (DEFAULT_LOAD).
    load_mw_reference = 68.0 * (float(DEFAULT_LOAD) / 3.0)

    result = run_physical_simulation_for_params(
        lat=43.4723,
        lon=-80.5449,
        load_mw=load_mw_reference,
        temp_c=DEFAULT_T_AIR,
        humidity=DEFAULT_HUMIDITY * 100.0,
        solar_wm2=DEFAULT_SOLAR_RADIATION,
        wind_speed_m_s=4.0,
        wind_direction="NE",
        wind_x=DEFAULT_WIND_X,
        wind_y=DEFAULT_WIND_Y,
        output_dir=output_dir,
        save_gif=save_gif,
        verbose=True,
    )

    metrics = result["metrics"]

    print("\n========================================")
    print("ONE PHYSICAL SIMULATION FINISHED")
    print("========================================")
    print(f"Output directory: {output_dir.resolve()}")

    print(f"Ambient temperature [C]: {metrics['T_air_C']:.3f}")
    print(f"Weather dissipation factor: {metrics['weather_dissipation_factor']:.3f}")
    print(
        "Datacenter cooling overhead factor: "
        f"{metrics['datacenter_cooling_overhead_factor']:.3f}"
    )
    print(f"Air density [kg/m3]: {metrics['air_density_kg_m3']:.3f}")
    print(f"Air areal heat capacity [J/m2K]: {metrics['air_areal_heat_capacity_J_m2K']:.3f}")

    print(f"Simulation time [s]: {metrics['simulation_time_s']:.2f}")
    print(f"Simulation time [min]: {metrics['simulation_time_min']:.2f}")
    print(f"dx [m]: {metrics['dx_m']:.3f}")
    print(f"dt [s]: {metrics['dt_s']:.3f}")
    print(f"Stable dt limit [s]: {metrics['stable_dt_limit_s']:.3f}")

    print(f"Building cells: {metrics['building_cells']}")
    print(f"Central building cells: {metrics['central_building_cells']}")

    print(f"Central building temperature [C]: {metrics['central_building_temperature_C']:.3f}")
    print(f"Central building delta above ambient [C]: {metrics['central_building_anomaly_C']:.3f}")

    print(f"Max temperature [C]: {metrics['max_temp_C']:.3f}")
    print(f"Mean temperature [C]: {metrics['mean_temp_C']:.3f}")
    print(f"Max delta above ambient [C]: {metrics['max_anomaly_C']:.3f}")
    print(f"Mean delta above ambient [C]: {metrics['mean_anomaly_C']:.3f}")

    print(f"Hot area > 0.2C delta [cells]: {metrics['hot_area_gt_0p2C_cells']}")
    print(f"Hot area > 0.5C delta [cells]: {metrics['hot_area_gt_0p5C_cells']}")
    print(f"Hot area > 1C delta [cells]: {metrics['hot_area_gt_1C_cells']}")
    print(f"Hot area > 2C delta [cells]: {metrics['hot_area_gt_2C_cells']}")
    print(f"Hot area > 5C delta [cells]: {metrics['hot_area_gt_5C_cells']}")

    print(f"Hot area > 30C absolute [cells]: {metrics['hot_area_gt_30C_cells']}")
    print(f"Hot area > 35C absolute [cells]: {metrics['hot_area_gt_35C_cells']}")
    print(f"Hot area > 40C absolute [cells]: {metrics['hot_area_gt_40C_cells']}")
    print(f"Hot area > 45C absolute [cells]: {metrics['hot_area_gt_45C_cells']}")
    print(f"Thermal risk objective: {metrics['thermal_risk_objective']:.3f}")

    print(f"Source heat density [W/m2]: {metrics['source_heat_density_W_m2']:.3f}")
    print(
        "Effective source heat density [W/m2]: "
        f"{metrics['effective_source_heat_density_W_m2']:.3f}"
    )
    print(f"Central solar heat density [W/m2]: {metrics['central_solar_heat_density_W_m2']:.3f}")

    if show_final_plot:
        T_final_v = np.load(output_dir / "temperature_final_C.npy")
        buildings_v = np.load(output_dir / "buildings_mask.npy")
        central_v = np.load(output_dir / "central_building_mask.npy")
        T_air_v = float(metrics["T_air_C"])

        plt.figure(figsize=(7, 6))

        plt.imshow(
            T_final_v.T,
            origin="lower",
            cmap="inferno",
            vmin=T_air_v,
        )

        plt.contour(
            buildings_v.T,
            levels=[0.5],
            colors="cyan",
            linewidths=0.25,
        )

        plt.contour(
            central_v.T,
            levels=[0.5],
            colors="white",
            linewidths=1.4,
        )

        plt.colorbar(label="Temperature [°C]")
        plt.title(f"Final temperature field, ambient = {T_air_v:.1f} °C")
        plt.xlabel("x pixel")
        plt.ylabel("y pixel")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()