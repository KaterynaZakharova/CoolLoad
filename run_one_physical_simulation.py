"""
Run one stable urban heat-transfer physical simulation around a central building.

Main behavior:
    - only the central building receives datacenter rejected heat
    - IT heat is linear in load_mw
    - chiller COP is estimated from a Carnot-style temperature lift
    - PUE = 1 + 1/COP + electrical/fan/pump overhead
    - facility heat rejected to the city is Q_rejected_to_outdoor_MW = load_mw * PUE
    - no artificial weather_dissipation_factor
    - no exponential load heat generation
    - no whole-field background heating
    - no periodic np.roll artifacts
    - outdoor heat loss uses physical convection and incremental longwave radiation
    - horizontal heat spreading uses a wind/mixing-length eddy diffusivity
    - transport wind and convective-cooling wind are separated so stronger wind
      removes heat instead of only spreading the plume

Required:
    from buildings_latlon import latlon_to_central_building_mask

Expected returned dictionary from latlon_to_central_building_mask:
    result["mask"]              -> full city buildings mask
    result["central_building"]  -> central building mask / heat source

Run:
    python run_one_physical_simulation.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from buildings_latlon import latlon_to_central_building_mask


# ============================================================
# 1. DEFAULT SIMULATION PARAMETERS
# ============================================================

DEFAULT_NX = 160
DEFAULT_NY = 160

# Keep this consistent with your current uploaded code.
# Change to 800.0 if you want the larger neighborhood.
DEFAULT_N_METERS = 400.0
DEFAULT_DX = DEFAULT_N_METERS / DEFAULT_NX
DEFAULT_DY = DEFAULT_N_METERS / DEFAULT_NY

DEFAULT_DT = 0.2
DEFAULT_STEPS = 3000
DEFAULT_SAVE_EVERY = 25

DEFAULT_T_AIR = 6.0
DEFAULT_HUMIDITY = 0.60
DEFAULT_SOLAR_RADIATION = 650.0

# Dimensionless fallback if load_mw is not provided.
DEFAULT_LOAD = 3.0
DEFAULT_IT_LOAD_REFERENCE_MW = 50.0

DEFAULT_WIND_X = 0.35
DEFAULT_WIND_Y = 0.16

MIXING_HEIGHT = 18.0

BUILDING_HEAT_CAPACITY_FACTOR = 2.5
SOURCE_BUILDING_HEAT_CAPACITY_FACTOR = 2.2

BUILDING_WIND_REDUCTION = 0.35
SOURCE_BUILDING_WIND_REDUCTION = 0.05

# Wind has two physical roles. Transport/advection can be strongly blocked by
# buildings, but convective cooling on exposed roofs/walls should still feel
# most of the ambient wind. If we use the strongly reduced transport wind for
# convection too, stronger wind can look falsely worse because it only spreads
# heat without removing enough heat from the hot source.
BUILDING_CONVECTIVE_WIND_REDUCTION = 0.65
SOURCE_BUILDING_CONVECTIVE_WIND_REDUCTION = 0.85

# Prevent unrealistically huge W/m² if the rasterized central building occupies
# only a few pixels. Facility heat rejection should be distributed over at least
# the physical roof/footprint scale.
USE_PHYSICAL_FOOTPRINT_FOR_SOURCE_AREA = True

# Solar is applied only to the central building/source roof.
CENTRAL_BUILDING_SOLAR_ABSORPTION = 0.55
CENTRAL_SOLAR_TO_HEAT_FRACTION = 0.012

# Calibration multiplier on W/m² source derived from MW rejected heat.
# Keep 1.0 for physical energy accounting.
SOURCE_HEAT_MULTIPLIER = 1.0

MAX_ALLOWED_TEMPERATURE_C = 85.0

# Radiation constants.
STEFAN_BOLTZMANN = 5.670374419e-8
URBAN_EFFECTIVE_EMISSIVITY = 0.90

# Moist air constants.
R_DRY_AIR = 287.058
R_WATER_VAPOR = 461.495
CP_DRY_AIR = 1005.0
CP_WATER_VAPOR = 1860.0
P_ATM = 101_325.0

# Horizontal thermal mixing.
MOLECULAR_THERMAL_DIFFUSIVITY_AIR = 2.2e-5
URBAN_EDDY_DIFFUSIVITY_COEFF = 0.08
MIN_URBAN_EDDY_DIFFUSIVITY = 0.15
BUILDING_DIFFUSIVITY_FACTOR = 0.20
SOURCE_BUILDING_DIFFUSIVITY_FACTOR = 0.35

# Exposure fractions for convective/radiative exchange.
STREET_EXPOSURE_FRACTION = 1.00
BUILDING_EXPOSURE_FRACTION = 0.35
SOURCE_BUILDING_EXPOSURE_FRACTION = 0.70

# Datacenter cooling model.
DC_INDOOR_SETPOINT_C = 24.0
CHILLED_WATER_SUPPLY_C = 12.0
CHILLER_CONDENSER_APPROACH_K = 8.0
CHILLER_EVAPORATOR_APPROACH_K = 5.0
CHILLER_CARNOT_EFFICIENCY = 0.35
POWER_DISTRIBUTION_LOSS_FACTOR = 0.06
FAN_PUMP_AUX_FACTOR = 0.04

U_ROOF_W_M2K = 0.35
VENTILATION_ACH = 0.5


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
# 2a. WALL / SLAB THERMAL OVERRIDES (volumetric heat capacity)
# ============================================================

_WALL_DEFAULT_DENSITY_KG_M3: dict[str, float] = {
    "concrete": 2400.0,
    "brick": 1800.0,
    "steel": 7850.0,
    "wood": 600.0,
    "glass": 2500.0,
    "granite": 2700.0,
    "gypsum": 1200.0,
    "stone": 2500.0,
    "default": 2400.0,
}


def default_density_for_wall_material(wall_material: str | None) -> float:
    if not wall_material:
        return _WALL_DEFAULT_DENSITY_KG_M3["default"]
    n = str(wall_material).lower()
    for key, rho in _WALL_DEFAULT_DENSITY_KG_M3.items():
        if key == "default":
            continue
        if key in n:
            return rho
    return _WALL_DEFAULT_DENSITY_KG_M3["default"]


def volumetric_heat_capacity_J_m3K(cp_kj_kg_k: float, density_kg_m3: float) -> float:
    return float(cp_kj_kg_k) * 1000.0 * float(density_kg_m3)


def merge_wall_thermal_into_datacenter_specs(
    base_specs: Dict[str, Any],
    *,
    wall_specific_heat_kj_per_kg_k: float,
    wall_density_kg_m3: float | None = None,
    wall_material: str | None = None,
) -> Dict[str, Any]:
    """
    Replace ``concrete_Cv_J_m3K`` used for opaque walls + slab thermal mass in
    ``compute_datacenter_effective_properties`` with rho*cp from the selected envelope solid.
    """
    specs = dict(base_specs)
    rho = (
        float(wall_density_kg_m3)
        if wall_density_kg_m3 is not None
        else default_density_for_wall_material(wall_material)
    )
    cv = volumetric_heat_capacity_J_m3K(wall_specific_heat_kj_per_kg_k, rho)
    specs["concrete_Cv_J_m3K"] = cv
    label = wall_material or "custom solid"
    specs["building_type"] = (
        f"{label} envelope (Cp={wall_specific_heat_kj_per_kg_k} kJ/kg·K, "
        f"rho={rho:.0f} kg/m3 -> Cv={cv:.3e} J/m3·K)"
    )
    return specs


# ============================================================
# 3. DATACENTER GEOMETRY / ENVELOPE HELPERS
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

    padded = np.pad(
        floor_plan,
        pad_width=1,
        mode="constant",
        constant_values=False,
    )

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
    heat_rejection_fraction = 1.00

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


def envelope_surfaces_and_areas(
    dc_props: Dict[str, Any],
    datacenter_specs: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], float, float, float, float]:
    U_wall = float(datacenter_specs["U_wall_W_m2K"])
    U_window = float(datacenter_specs["U_window_W_m2K"])

    A_opaque = float(dc_props["opaque_wall_area_m2"])
    A_glass = float(dc_props["glass_area_m2"])
    footprint = float(dc_props["footprint_area_m2"])

    surfaces: List[Dict[str, Any]] = [
        {"name": "opaque_wall", "U_W_m2K": U_wall, "A_m2": A_opaque},
        {"name": "windows", "U_W_m2K": U_window, "A_m2": A_glass},
        {"name": "roof", "U_W_m2K": U_ROOF_W_M2K, "A_m2": footprint},
    ]

    A_exposed_m2 = A_opaque + A_glass + footprint
    A_solar_m2 = footprint
    A_window_solar_m2 = A_glass

    num_floors = max(int(datacenter_specs["num_floors"]), 1)
    floor_h = float(datacenter_specs["floor_height_m"])
    volume_m3 = max(footprint * num_floors * floor_h, 1.0)

    Vdot_m3_s = VENTILATION_ACH * volume_m3 / 3600.0

    return surfaces, A_exposed_m2, A_solar_m2, A_window_solar_m2, Vdot_m3_s


# ============================================================
# 4. PHYSICAL WEATHER / AIR / HEAT TRANSFER
# ============================================================

def _humidity_fraction(humidity: float) -> float:
    h = float(humidity)
    if h > 1.0 + 1e-6:
        h /= 100.0
    return float(np.clip(h, 0.0, 1.0))


def saturation_vapor_pressure_Pa(T_C: float) -> float:
    T = float(T_C)
    return 611.21 * math.exp((18.678 - T / 234.5) * (T / (257.14 + T)))


def vapor_pressure_Pa(T_C: float, relative_humidity: float) -> float:
    rh = _humidity_fraction(relative_humidity)
    return rh * saturation_vapor_pressure_Pa(T_C)


def moist_air_density_kg_m3(
    T_C: float,
    relative_humidity: float,
    pressure_Pa: float = P_ATM,
) -> float:
    T_K = float(T_C) + 273.15
    p_v = vapor_pressure_Pa(T_C, relative_humidity)
    p_d = pressure_Pa - p_v

    rho = p_d / (R_DRY_AIR * T_K) + p_v / (R_WATER_VAPOR * T_K)
    return float(rho)


def moist_air_cp_J_kgK(
    T_C: float,
    relative_humidity: float,
    pressure_Pa: float = P_ATM,
) -> float:
    p_v = vapor_pressure_Pa(T_C, relative_humidity)
    w = 0.62198 * p_v / max(pressure_Pa - p_v, 1.0)

    cp = (CP_DRY_AIR + w * CP_WATER_VAPOR) / (1.0 + w)
    return float(cp)


def air_areal_heat_capacity(
    T_air_C: float,
    relative_humidity: float,
    mixing_height_m: float = MIXING_HEIGHT,
) -> float:
    rho = moist_air_density_kg_m3(T_air_C, relative_humidity)
    cp = moist_air_cp_J_kgK(T_air_C, relative_humidity)
    return float(rho * cp * mixing_height_m)


def external_convection_h_W_m2K(
    wind_speed_m_s: np.ndarray | float,
) -> np.ndarray | float:
    v = np.maximum(wind_speed_m_s, 0.0)
    return 5.7 + 3.8 * v


def sky_emissivity_clear(
    T_air_C: float,
    relative_humidity: float,
) -> float:
    e_Pa = vapor_pressure_Pa(T_air_C, relative_humidity)
    e_hPa = max(e_Pa / 100.0, 1e-6)

    eps = 0.618 + 0.056 * math.sqrt(e_hPa)
    return float(np.clip(eps, 0.55, 1.0))


def effective_sky_temperature_C(
    T_air_C: float,
    relative_humidity: float,
    cloud_fraction: float = 0.0,
) -> float:
    T_air_K = float(T_air_C) + 273.15

    eps_clear = sky_emissivity_clear(T_air_C, relative_humidity)
    cloud = float(np.clip(cloud_fraction, 0.0, 1.0))

    eps_sky = eps_clear * (1.0 - cloud) + 1.0 * cloud
    eps_sky = float(np.clip(eps_sky, 0.55, 1.0))

    T_sky_K = (eps_sky ** 0.25) * T_air_K
    return float(T_sky_K - 273.15)


def exposure_fraction_field(
    buildings: np.ndarray,
    central_building_mask: np.ndarray,
) -> np.ndarray:
    exposure = np.where(
        buildings,
        BUILDING_EXPOSURE_FRACTION,
        STREET_EXPOSURE_FRACTION,
    )

    exposure = np.where(
        central_building_mask,
        SOURCE_BUILDING_EXPOSURE_FRACTION,
        exposure,
    )

    return exposure.astype(float)


def convective_heat_loss_W_m2(
    T: np.ndarray,
    T_air_C: float,
    wind_speed: np.ndarray,
    exposure_fraction: np.ndarray,
) -> np.ndarray:
    h = external_convection_h_W_m2K(wind_speed)
    return exposure_fraction * h * (T - T_air_C)


def radiative_heat_loss_W_m2(
    T: np.ndarray,
    T_air_C: float,
    relative_humidity: float,
    exposure_fraction: np.ndarray,
    emissivity: float = URBAN_EFFECTIVE_EMISSIVITY,
) -> np.ndarray:
    """
    Incremental longwave cooling relative to the ambient equilibrium.

    This avoids cooling the whole domain below T_air just because the sky is
    radiatively colder. For an anomaly PDE, the relevant term is:

        epsilon sigma (T_cell^4 - T_air^4)

    not the full surface-sky balance.
    """
    T_K = np.maximum(T + 273.15, 1.0)
    T_air_K = float(T_air_C) + 273.15

    return (
        exposure_fraction
        * emissivity
        * STEFAN_BOLTZMANN
        * (T_K**4 - T_air_K**4)
    )


def urban_eddy_diffusivity_field(
    buildings: np.ndarray,
    central_building_mask: np.ndarray,
    wind_speed: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    L = math.sqrt(float(dx) * float(dy))

    K_eddy = URBAN_EDDY_DIFFUSIVITY_COEFF * np.maximum(wind_speed, 0.0) * L

    alpha = MOLECULAR_THERMAL_DIFFUSIVITY_AIR + np.maximum(
        K_eddy,
        MIN_URBAN_EDDY_DIFFUSIVITY,
    )

    alpha = np.where(buildings, alpha * BUILDING_DIFFUSIVITY_FACTOR, alpha)
    alpha = np.where(
        central_building_mask,
        alpha * SOURCE_BUILDING_DIFFUSIVITY_FACTOR,
        alpha,
    )

    return alpha.astype(float)


# ============================================================
# 5. FACILITY HEAT / COP / PUE / PASSIVE REPORTING
# ============================================================

def chiller_COP_from_temperatures(
    T_chilled_water_C: float,
    T_outdoor_C: float,
    condenser_approach_K: float = CHILLER_CONDENSER_APPROACH_K,
    evaporator_approach_K: float = CHILLER_EVAPORATOR_APPROACH_K,
    carnot_efficiency: float = CHILLER_CARNOT_EFFICIENCY,
    COP_min: float = 1.2,
    COP_max: float = 9.0,
) -> float:
    T_cold_K = float(T_chilled_water_C) + 273.15 - evaporator_approach_K
    T_hot_K = float(T_outdoor_C) + 273.15 + condenser_approach_K

    delta_T = max(T_hot_K - T_cold_K, 1.0)

    COP_carnot = T_cold_K / delta_T
    COP_real = carnot_efficiency * COP_carnot

    return float(np.clip(COP_real, COP_min, COP_max))


def pue_from_COP(
    COP: float,
    power_distribution_loss_factor: float = POWER_DISTRIBUTION_LOSS_FACTOR,
    fan_pump_aux_factor: float = FAN_PUMP_AUX_FACTOR,
) -> float:
    COP = max(float(COP), 1e-6)

    pue = (
        1.0
        + 1.0 / COP
        + power_distribution_loss_factor
        + fan_pump_aux_factor
    )

    return float(np.clip(pue, 1.03, 2.0))


def datacenter_facility_heat_MW(
    *,
    load_mw: float,
    T_outdoor_C: float,
    T_chilled_water_C: float = CHILLED_WATER_SUPPLY_C,
) -> Dict[str, float]:
    Q_it_MW = max(float(load_mw), 0.0)

    COP = chiller_COP_from_temperatures(
        T_chilled_water_C=T_chilled_water_C,
        T_outdoor_C=T_outdoor_C,
    )

    PUE = pue_from_COP(COP)

    Q_cooling_electric_MW = Q_it_MW / COP
    Q_facility_MW = Q_it_MW * PUE

    return {
        "Q_it_MW": float(Q_it_MW),
        "COP": float(COP),
        "PUE": float(PUE),
        "Q_cooling_electric_MW": float(Q_cooling_electric_MW),
        "Q_facility_MW": float(Q_facility_MW),
        "Q_rejected_to_outdoor_MW": float(Q_facility_MW),
    }


def conduction_loss_MW(
    surfaces: List[Dict[str, Any]],
    T_inside_C: float,
    T_out_C: float,
) -> float:
    q_W = 0.0

    for s in surfaces:
        q_W += (
            float(s["U_W_m2K"])
            * float(s["A_m2"])
            * (float(T_inside_C) - float(T_out_C))
        )

    return q_W / 1e6


def ventilation_loss_MW(
    Vdot_m3_s: float,
    T_inside_C: float,
    T_out_C: float,
    relative_humidity: float,
) -> float:
    rho = moist_air_density_kg_m3(T_out_C, relative_humidity)
    cp = moist_air_cp_J_kgK(T_out_C, relative_humidity)

    return (
        rho
        * cp
        * float(Vdot_m3_s)
        * (float(T_inside_C) - float(T_out_C))
        / 1e6
    )


def passive_envelope_terms_MW(
    *,
    T_inside_C: float,
    T_out_C: float,
    T_surface_C: float,
    relative_humidity: float,
    wind_speed_m_s: float,
    solar_wm2: float,
    surfaces: List[Dict[str, Any]],
    A_exposed_m2: float,
    A_solar_m2: float,
    A_window_solar_m2: float,
    Vdot_m3_s: float,
    emissivity: float = 0.90,
    absorptivity: float = 0.60,
    SHGC: float = 0.35,
) -> Dict[str, float]:
    Q_cond_loss_MW = conduction_loss_MW(
        surfaces=surfaces,
        T_inside_C=T_inside_C,
        T_out_C=T_out_C,
    )

    Q_vent_loss_MW = ventilation_loss_MW(
        Vdot_m3_s=Vdot_m3_s,
        T_inside_C=T_inside_C,
        T_out_C=T_out_C,
        relative_humidity=relative_humidity,
    )

    h = external_convection_h_W_m2K(wind_speed_m_s)
    Q_conv_loss_MW = (
        h * A_exposed_m2 * (T_surface_C - T_out_C) / 1e6
    )

    T_sky_C = effective_sky_temperature_C(
        T_air_C=T_out_C,
        relative_humidity=relative_humidity,
    )

    T_surface_K = T_surface_C + 273.15
    T_sky_K = T_sky_C + 273.15

    Q_rad_loss_MW = (
        emissivity
        * STEFAN_BOLTZMANN
        * A_exposed_m2
        * (T_surface_K**4 - T_sky_K**4)
        / 1e6
    )

    Q_solar_opaque_MW = absorptivity * A_solar_m2 * solar_wm2 / 1e6
    Q_solar_window_MW = SHGC * A_window_solar_m2 * solar_wm2 / 1e6
    Q_solar_gain_MW = Q_solar_opaque_MW + Q_solar_window_MW

    Q_passive_net_MW = (
        Q_solar_gain_MW
        - Q_cond_loss_MW
        - Q_vent_loss_MW
        - Q_conv_loss_MW
        - Q_rad_loss_MW
    )

    return {
        "Q_solar_opaque_MW": float(Q_solar_opaque_MW),
        "Q_solar_window_MW": float(Q_solar_window_MW),
        "Q_solar_gain_MW": float(Q_solar_gain_MW),
        "Q_conduction_loss_MW": float(Q_cond_loss_MW),
        "Q_ventilation_loss_MW": float(Q_vent_loss_MW),
        "Q_convection_loss_MW": float(Q_conv_loss_MW),
        "Q_radiation_loss_MW": float(Q_rad_loss_MW),
        "Q_passive_net_MW": float(Q_passive_net_MW),
        "T_sky_C": float(T_sky_C),
        "h_external_W_m2K": float(h),
    }


def datacenter_heat_terms_MW(
    *,
    load_mw: float,
    T_inside_C: float,
    T_out_C: float,
    T_surface_C: float,
    relative_humidity: float,
    wind_speed_m_s: float,
    solar_wm2: float,
    surfaces: List[Dict[str, Any]],
    A_exposed_m2: float,
    A_solar_m2: float,
    Vdot_m3_s: float,
    A_window_solar_m2: float = 0.0,
) -> Dict[str, float]:
    facility = datacenter_facility_heat_MW(
        load_mw=load_mw,
        T_outdoor_C=T_out_C,
    )

    passive = passive_envelope_terms_MW(
        T_inside_C=T_inside_C,
        T_out_C=T_out_C,
        T_surface_C=T_surface_C,
        relative_humidity=relative_humidity,
        wind_speed_m_s=wind_speed_m_s,
        solar_wm2=solar_wm2,
        surfaces=surfaces,
        A_exposed_m2=A_exposed_m2,
        A_solar_m2=A_solar_m2,
        A_window_solar_m2=A_window_solar_m2,
        Vdot_m3_s=Vdot_m3_s,
    )

    return {**facility, **passive}


# ============================================================
# 6. MASK UTILITIES
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
# 7. STABLE NUMERICAL OPERATORS
# ============================================================

def neighbor_arrays_edge(
    T: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    T[0, :] = 0.6 * T[1, :] + 0.4 * T_air
    T[-1, :] = 0.6 * T[-2, :] + 0.4 * T_air
    T[:, 0] = 0.6 * T[:, 1] + 0.4 * T_air
    T[:, -1] = 0.6 * T[:, -2] + 0.4 * T_air

    return T


def enforce_uniform_central_building_temperature(
    T: np.ndarray,
    central_building_mask: np.ndarray,
) -> np.ndarray:
    if central_building_mask.sum() == 0:
        return T

    central_temp = float(T[central_building_mask].mean())
    T[central_building_mask] = central_temp

    return T


def local_smooth_anomaly(
    T: np.ndarray,
    T_air: float,
    central_building_mask: np.ndarray,
    strength: float = 0.025,
) -> np.ndarray:
    """
    Weak numerical filter for checkerboard damping.

    This is not a physical source/sink. It only smooths grid-scale noise in
    the anomaly field and is kept deliberately small.
    """
    anomaly = T - T_air

    A_ip, A_im, A_jp, A_jm = neighbor_arrays_edge(anomaly)
    neighbor_mean = 0.25 * (A_ip + A_im + A_jp + A_jm)

    anomaly = (1.0 - strength) * anomaly + strength * neighbor_mean

    T_new = T_air + anomaly
    T_new = enforce_uniform_central_building_temperature(
        T_new,
        central_building_mask,
    )

    return T_new


# ============================================================
# 8. MAIN PHYSICAL MODEL
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
    wind_speed_m_s: float = 5.0,
    load_mw: float | None = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:

    if datacenter_specs is None:
        datacenter_specs = DATACENTER_SPECS

    humidity = _humidity_fraction(humidity)

    buildings = buildings.astype(bool)
    central_building_mask = central_building_mask.astype(bool)

    buildings = buildings | central_building_mask
    central_building_mask = central_building_mask & buildings

    if central_building_mask.sum() == 0:
        raise ValueError("central_building_mask is empty.")

    nx, ny = buildings.shape

    dc_props = compute_datacenter_effective_properties(datacenter_specs)

    T = np.ones((nx, ny), dtype=float) * T_air

    source_mask = central_building_mask.astype(float)
    source_cell_count = max(int(central_building_mask.sum()), 1)

    C_air = air_areal_heat_capacity(
        T_air_C=T_air,
        relative_humidity=humidity,
    )

    load_mw_eff = (
        float(load_mw)
        if load_mw is not None
        else float(load) * DEFAULT_IT_LOAD_REFERENCE_MW
    )
    load_mw_eff = max(load_mw_eff, 0.0)

    surfaces, A_exposed_m2, A_solar_m2, A_window_solar_m2, Vdot_m3_s = (
        envelope_surfaces_and_areas(dc_props, datacenter_specs)
    )

    cell_area_m2 = float(dx) * float(dy)
    mask_source_area_m2 = float(source_cell_count) * cell_area_m2
    physical_roof_area_m2 = float(dc_props["footprint_area_m2"])

    if USE_PHYSICAL_FOOTPRINT_FOR_SOURCE_AREA:
        source_area_m2 = max(mask_source_area_m2, physical_roof_area_m2)
    else:
        source_area_m2 = mask_source_area_m2

    T_inside_c = DC_INDOOR_SETPOINT_C
    ws_m = float(max(wind_speed_m_s, 0.0))

    T_surface_guess = max(float(T_air), float(T_inside_c) - 2.0)

    heat_terms_run = datacenter_heat_terms_MW(
        load_mw=load_mw_eff,
        T_inside_C=T_inside_c,
        T_out_C=float(T_air),
        T_surface_C=T_surface_guess,
        relative_humidity=float(humidity),
        wind_speed_m_s=ws_m,
        solar_wm2=float(solar_radiation),
        surfaces=surfaces,
        A_exposed_m2=A_exposed_m2,
        A_solar_m2=A_solar_m2,
        Vdot_m3_s=Vdot_m3_s,
        A_window_solar_m2=A_window_solar_m2,
    )

    Q_rejected_MW = heat_terms_run["Q_rejected_to_outdoor_MW"]
    Q_source_W = Q_rejected_MW * 1e6
    Q_source_W_m2 = (
        SOURCE_HEAT_MULTIPLIER
        * Q_source_W
        / max(source_area_m2, 1e-9)
    )

    Q_central_solar_W_m2 = (
        solar_radiation
        * CENTRAL_BUILDING_SOLAR_ABSORPTION
        * CENTRAL_SOLAR_TO_HEAT_FRACTION
    )

    Q_source = (
        Q_source_W_m2 * source_mask
        + Q_central_solar_W_m2 * source_mask
    )

    # ------------------------------------------------------------
    # Wind fields: transport wind vs convective-cooling wind
    # ------------------------------------------------------------
    # Transport/advection wind is strongly blocked by buildings.
    transport_wind_reduction = np.where(buildings, BUILDING_WIND_REDUCTION, 1.0)
    transport_wind_reduction = np.where(
        central_building_mask,
        SOURCE_BUILDING_WIND_REDUCTION,
        transport_wind_reduction,
    )

    u = wind_x * transport_wind_reduction
    v = wind_y * transport_wind_reduction
    transport_wind_speed = np.sqrt(u**2 + v**2)

    # Convective cooling wind should not be reduced as aggressively as the
    # street-level transport wind. Exposed walls, roofs, and exhaust plumes
    # still exchange heat with outdoor flow, so stronger ambient wind should
    # increase heat removal even if urban geometry blocks part of the advection.
    ambient_wind_speed = float(max(wind_speed_m_s, 0.0))
    convective_wind_speed = np.ones_like(transport_wind_speed) * ambient_wind_speed
    convective_wind_speed = np.where(
        buildings,
        BUILDING_CONVECTIVE_WIND_REDUCTION * ambient_wind_speed,
        convective_wind_speed,
    )
    convective_wind_speed = np.where(
        central_building_mask,
        SOURCE_BUILDING_CONVECTIVE_WIND_REDUCTION * ambient_wind_speed,
        convective_wind_speed,
    )

    alpha = urban_eddy_diffusivity_field(
        buildings=buildings,
        central_building_mask=central_building_mask,
        wind_speed=transport_wind_speed,
        dx=dx,
        dy=dy,
    )

    exposure_fraction = exposure_fraction_field(
        buildings=buildings,
        central_building_mask=central_building_mask,
    )

    alpha_max = float(alpha.max())
    stable_dt_limit = 0.24 * min(dx, dy) ** 2 / max(alpha_max, 1e-12)

    if dt > stable_dt_limit:
        raise ValueError(
            f"Unstable explicit diffusion setting: dt={dt:.4f}, "
            f"stable limit is about {stable_dt_limit:.4f}. "
            "Reduce DEFAULT_DT or the eddy diffusivity coefficient."
        )

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

    states = []

    T = enforce_uniform_central_building_temperature(
        T,
        central_building_mask,
    )

    for n in range(steps):
        dTdx, dTdy = upwind_gradient(T, u, v, dx, dy)
        dTdt_adv = -(u * dTdx + v * dTdy)

        dTdt_diff = variable_diffusion(T, alpha, dx, dy)

        dTdt_source = Q_source / C_eff

        q_conv = convective_heat_loss_W_m2(
            T=T,
            T_air_C=T_air,
            wind_speed=convective_wind_speed,
            exposure_fraction=exposure_fraction,
        )
        dTdt_conv = -q_conv / C_eff

        q_rad = radiative_heat_loss_W_m2(
            T=T,
            T_air_C=T_air,
            relative_humidity=humidity,
            exposure_fraction=exposure_fraction,
        )
        dTdt_rad = -q_rad / C_eff

        dTdt = (
            dTdt_adv
            + dTdt_diff
            + dTdt_source
            + dTdt_conv
            + dTdt_rad
        )

        T = T + dt * dTdt

        T = enforce_uniform_central_building_temperature(
            T,
            central_building_mask,
        )

        if n % 4 == 0:
            T = local_smooth_anomaly(
                T=T,
                T_air=T_air,
                central_building_mask=central_building_mask,
                strength=0.025,
            )

        T = apply_open_boundary(T, T_air)

        T = np.clip(T, T_air - 2.0, MAX_ALLOWED_TEMPERATURE_C)

        T = enforce_uniform_central_building_temperature(
            T,
            central_building_mask,
        )

        if return_states and n % save_every == 0:
            states.append(T.copy())

    anomaly = T - T_air

    central_temp = float(T[central_building_mask].mean())
    central_anomaly = central_temp - T_air

    heat_terms_final = datacenter_heat_terms_MW(
        load_mw=load_mw_eff,
        T_inside_C=T_inside_c,
        T_out_C=float(T_air),
        T_surface_C=central_temp,
        relative_humidity=float(humidity),
        wind_speed_m_s=ws_m,
        solar_wm2=float(solar_radiation),
        surfaces=surfaces,
        A_exposed_m2=A_exposed_m2,
        A_solar_m2=A_solar_m2,
        Vdot_m3_s=Vdot_m3_s,
        A_window_solar_m2=A_window_solar_m2,
    )

    thermal_risk_objective = float(
        1.0 * float(T.max())
        + 0.01 * int(np.sum(T > 35.0))
        + 0.03 * int(np.sum(T > 40.0))
        + 0.10 * int(np.sum(T > 45.0))
    )

    metrics: Dict[str, Any] = {
        "load": float(load),
        "load_mw": float(load_mw_eff),

        "Q_source_W_m2": float(Q_source_W_m2),
        "Q_central_solar_W_m2": float(Q_central_solar_W_m2),
        "source_heat_multiplier": float(SOURCE_HEAT_MULTIPLIER),

        "wind_x_m_s": float(wind_x),
        "wind_y_m_s": float(wind_y),
        "ambient_wind_speed_m_s": float(ws_m),
        "mean_transport_wind_speed_m_s": float(np.mean(transport_wind_speed)),
        "mean_convective_wind_speed_m_s": float(np.mean(convective_wind_speed)),

        "T_air_C": float(T_air),
        "humidity": float(humidity),
        "solar_radiation_W_m2": float(solar_radiation),

        "pue_nominal": float(dc_props["pue"]),
        "air_density_kg_m3": float(moist_air_density_kg_m3(T_air, humidity)),
        "air_cp_J_kgK": float(moist_air_cp_J_kgK(T_air, humidity)),
        "air_areal_heat_capacity_J_m2K": float(C_air),

        "effective_sky_temperature_C": float(
            effective_sky_temperature_C(T_air, humidity)
        ),
        "mean_convective_h_W_m2K": float(
            np.mean(external_convection_h_W_m2K(convective_wind_speed))
        ),
        "mean_exposure_fraction": float(np.mean(exposure_fraction)),

        "dt_s": float(dt),
        "steps": int(steps),
        "simulation_time_s": float(dt * steps),
        "simulation_time_min": float(dt * steps / 60.0),

        "dx_m": float(dx),
        "dy_m": float(dy),
        "alpha_max_m2_s": float(alpha_max),
        "alpha_mean_m2_s": float(alpha.mean()),
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
        "mask_source_area_m2": float(mask_source_area_m2),
        "physical_roof_area_m2": float(physical_roof_area_m2),
        "source_area_m2": float(source_area_m2),

        "usable_power_density_W_m2": float(dc_props["usable_power_density_W_m2"]),
        "source_heat_density_W_m2": float(Q_source_W_m2),
        "effective_source_heat_density_W_m2": float(Q_source_W_m2),
        "central_solar_heat_density_W_m2": float(Q_central_solar_W_m2),

        "window_to_wall_ratio": float(dc_props["window_to_wall_ratio"]),
        "glass_area_m2": float(dc_props["glass_area_m2"]),
        "opaque_wall_area_m2": float(dc_props["opaque_wall_area_m2"]),
        "total_wall_area_m2": float(dc_props["total_wall_area_m2"]),
        "UA_total_W_K": float(dc_props["UA_total_W_K"]),
        "H_envelope_areal_W_m2K": float(dc_props["H_envelope_areal_W_m2K"]),
        "heat_rejection_fraction": float(dc_props["heat_rejection_fraction"]),
        "ventilation_Vdot_m3_s": float(Vdot_m3_s),
        "dc_indoor_setpoint_C": float(T_inside_c),
    }

    for key, val in heat_terms_final.items():
        metrics[key] = float(val)

    states_arr = np.array(states) if return_states else np.empty((0, nx, ny))

    return metrics, T, states_arr


# ============================================================
# 9. PLOTTING / SAVING
# ============================================================

def _style_clean_thermal_figure(fig, ax) -> None:
    ax.set_axis_off()
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)


def plot_masks(
    buildings: np.ndarray,
    central_building_mask: np.ndarray,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.5), facecolor="#0b1220")
    ax.set_facecolor("#0b1220")

    ax.imshow(buildings.T, origin="lower", cmap="bone")

    ax.contour(
        central_building_mask.T,
        levels=[0.5],
        colors="#f472b6",
        linewidths=1.8,
    )

    _style_clean_thermal_figure(fig, ax)
    fig.savefig(
        output_dir / "01_building_masks.png",
        dpi=200,
        facecolor=fig.get_facecolor(),
    )
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

    fig, ax = plt.subplots(figsize=(6.5, 6.5), facecolor="#0b1220")
    ax.set_facecolor("#0b1220")

    vmax_temp = max(float(np.percentile(T_final, 99.7)), T_air + 1.0)

    ax.imshow(
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
        linewidths=0.35,
        alpha=0.65,
    )

    ax.contour(
        central_building_mask.T,
        levels=[0.5],
        colors="white",
        linewidths=1.5,
    )

    _style_clean_thermal_figure(fig, ax)
    tmax = float(T_final.max())
    fig.text(
        0.04,
        0.94,
        f"Ambient {T_air:.1f}°C · peak {tmax:.1f}°C",
        color="white",
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#020617",
            edgecolor="white",
            alpha=0.75,
        ),
    )

    fig.savefig(
        output_dir / "02_final_temperature.png",
        dpi=200,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 6.5), facecolor="#0b1220")
    ax.set_facecolor("#0b1220")

    vmax_anomaly = max(float(np.percentile(anomaly, 99.7)), 0.5)

    ax.imshow(
        anomaly.T,
        origin="lower",
        cmap="RdYlBu_r",
        vmin=0.0,
        vmax=vmax_anomaly,
    )

    ax.contour(
        buildings.T,
        levels=[0.5],
        colors="white",
        linewidths=0.3,
        alpha=0.5,
    )

    ax.contour(
        central_building_mask.T,
        levels=[0.5],
        colors="white",
        linewidths=1.4,
    )

    _style_clean_thermal_figure(fig, ax)
    fig.text(
        0.04,
        0.94,
        f"ΔT above ambient (max {float(anomaly.max()):.2f}°C)",
        color="white",
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#020617",
            edgecolor="white",
            alpha=0.75,
        ),
    )

    fig.savefig(
        output_dir / "03_final_anomaly.png",
        dpi=200,
        facecolor=fig.get_facecolor(),
    )
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

    fig, ax = plt.subplots(figsize=(6.5, 6.5), facecolor="#0b1220")
    ax.set_facecolor("#0b1220")

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
        linewidths=0.28,
        alpha=0.55,
    )

    ax.contour(
        central_building_mask.T,
        levels=[0.5],
        colors="white",
        linewidths=1.2,
    )

    _style_clean_thermal_figure(fig, ax)
    title = fig.suptitle(
        f"Heat plume · {T_air:.1f}°C ambient",
        color="white",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    def update(frame: int):
        im.set_data(states[frame].T)

        minutes = frame * save_every * dt / 60.0
        title.set_text(f"Heat plume · {T_air:.1f}°C · t = {minutes:.1f} min")

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
            savefig_kwargs={"facecolor": fig.get_facecolor()},
        )
    except Exception as exc:
        print(f"Could not save GIF animation. Reason: {exc}")

    plt.close(fig)


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
# 10. UI / PIPELINE HELPERS
# ============================================================

def mw_it_load_multiplier(load_mw: float, reference_mw: float = DEFAULT_IT_LOAD_REFERENCE_MW) -> float:
    ref = max(float(reference_mw), 1.0)
    return float(max(load_mw, 0.0) / ref)


def wind_cardinal_to_components(
    wind_speed_m_s: float,
    wind_direction: str,
    scale: float = 0.1,
) -> Tuple[float, float]:
    """
    Convert UI wind speed and cardinal direction FROM which wind blows to
    model advection components.

    Example:
        wind_direction="N" means wind blows from north to south.
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
    datacenter_specs: Dict[str, Any] | None = None,
    wall_specific_heat_kj_per_kg_k: float | None = None,
    wall_density_kg_m3: float | None = None,
    wall_material: str | None = None,
) -> Dict[str, Any]:
    """
    Full pipeline:
        1. fetch building masks at lat/lon
        2. run heat PDE
        3. optionally write PNG/GIF/NumPy/JSON outputs

    Returns:
        {"metrics": metrics, "output_dir": output_dir}
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

    if datacenter_specs is not None:
        dc_specs_use = dict(datacenter_specs)
    elif wall_specific_heat_kj_per_kg_k is not None:
        dc_specs_use = merge_wall_thermal_into_datacenter_specs(
            DATACENTER_SPECS,
            wall_specific_heat_kj_per_kg_k=float(wall_specific_heat_kj_per_kg_k),
            wall_density_kg_m3=wall_density_kg_m3,
            wall_material=wall_material,
        )
    else:
        dc_specs_use = dict(DATACENTER_SPECS)

    metrics, T_final, states = simulate_city_heat_one_case(
        buildings=buildings,
        central_building_mask=central_building_mask,
        load=load,
        wind_x=wx,
        wind_y=wy,
        T_air=T_air,
        humidity=humidity_f,
        solar_radiation=solar_radiation,
        datacenter_specs=dc_specs_use,
        dx=dx,
        dy=dy,
        dt=dt,
        steps=steps,
        save_every=save_every,
        return_states=return_states,
        wind_speed_m_s=max(float(wind_speed_m_s), 0.0),
        load_mw=float(load_mw),
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


# ============================================================
# 11. MAIN
# ============================================================

def main() -> None:
    output_dir = Path("simulation_outputs")
    save_gif = True
    show_final_plot = False

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
    print(
        f"PUE: {metrics['PUE']:.3f} "
        f"(nominal design PUE {metrics['pue_nominal']:.3f}, "
        f"COP {metrics['COP']:.3f}, "
        f"Q_rejected {metrics['Q_rejected_to_outdoor_MW']:.4f} MW)"
    )
    print(f"Q_source facility [W/m²]: {metrics['Q_source_W_m2']:.3f}")
    print(f"Q_central_solar [W/m²]: {metrics['Q_central_solar_W_m2']:.3f}")

    print(f"Air density [kg/m3]: {metrics['air_density_kg_m3']:.3f}")
    print(f"Air cp [J/kgK]: {metrics['air_cp_J_kgK']:.3f}")
    print(f"Air areal heat capacity [J/m2K]: {metrics['air_areal_heat_capacity_J_m2K']:.3f}")
    print(f"Effective sky temperature [C]: {metrics['effective_sky_temperature_C']:.3f}")

    print(f"Mean transport wind [m/s]: {metrics['mean_transport_wind_speed_m_s']:.3f}")
    print(f"Mean convective wind [m/s]: {metrics['mean_convective_wind_speed_m_s']:.3f}")
    print(f"Mean convective h [W/m2K]: {metrics['mean_convective_h_W_m2K']:.3f}")
    print(f"Mean exposure fraction: {metrics['mean_exposure_fraction']:.3f}")
    print(f"Mean alpha [m2/s]: {metrics['alpha_mean_m2_s']:.4f}")
    print(f"Max alpha [m2/s]: {metrics['alpha_max_m2_s']:.4f}")

    print(f"Simulation time [s]: {metrics['simulation_time_s']:.2f}")
    print(f"Simulation time [min]: {metrics['simulation_time_min']:.2f}")
    print(f"dx [m]: {metrics['dx_m']:.3f}")
    print(f"dt [s]: {metrics['dt_s']:.3f}")
    print(f"Stable dt limit [s]: {metrics['stable_dt_limit_s']:.3f}")

    print(f"Building cells: {metrics['building_cells']}")
    print(f"Central building cells: {metrics['central_building_cells']}")
    print(f"Mask source area [m2]: {metrics['mask_source_area_m2']:.3f}")
    print(f"Physical roof area [m2]: {metrics['physical_roof_area_m2']:.3f}")
    print(f"Effective source area [m2]: {metrics['source_area_m2']:.3f}")

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

    print("\nPassive envelope/facility terms:")
    print(f"Q_it_MW: {metrics['Q_it_MW']:.4f}")
    print(f"Q_cooling_electric_MW: {metrics['Q_cooling_electric_MW']:.4f}")
    print(f"Q_facility_MW: {metrics['Q_facility_MW']:.4f}")
    print(f"Q_rejected_to_outdoor_MW: {metrics['Q_rejected_to_outdoor_MW']:.4f}")
    print(f"Q_solar_gain_MW: {metrics['Q_solar_gain_MW']:.4f}")
    print(f"Q_conduction_loss_MW: {metrics['Q_conduction_loss_MW']:.4f}")
    print(f"Q_ventilation_loss_MW: {metrics['Q_ventilation_loss_MW']:.4f}")
    print(f"Q_convection_loss_MW: {metrics['Q_convection_loss_MW']:.4f}")
    print(f"Q_radiation_loss_MW: {metrics['Q_radiation_loss_MW']:.4f}")
    print(f"Q_passive_net_MW: {metrics['Q_passive_net_MW']:.4f}")

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