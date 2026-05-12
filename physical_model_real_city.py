import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from buildings_latlon import latlon_to_central_building_mask


# ============================================================
# DEFAULT SIMULATION PARAMETERS
# ============================================================

DEFAULT_DT = 1.0
DEFAULT_STEPS = 1200
DEFAULT_SAVE_EVERY = 20

DEFAULT_T_AIR = 26.0
DEFAULT_HUMIDITY = 0.60
DEFAULT_SOLAR_RADIATION = 650.0  # W/m2

DEFAULT_RHO_AIR = 1.2
DEFAULT_CP_AIR = 1005.0
DEFAULT_MIXING_HEIGHT = 18.0

# Effective heat diffusion in urban air / buildings.
# These are effective urban-scale parameters, not molecular diffusivities.
DEFAULT_ALPHA_AIR = 18.0
DEFAULT_ALPHA_BUILDING = 2.0
DEFAULT_ALPHA_INTERFACE = 5.0

DEFAULT_BASE_COOLING_RATE = 3.0e-4

DEFAULT_BUILDING_HEAT_CAPACITY_FACTOR = 3.0

DEFAULT_LOAD_GAMMA = 1.65

DEFAULT_MAX_TEMP_CLIP = 55.0


# ============================================================
# DATACENTER SPECS
# ============================================================

DEFAULT_DATACENTER_SPECS = {
    "colocation_area_ft2": 108_976.0,
    "colocation_area_m2": 10_124.0,
    "num_floors": 4,
    "footprint_area_m2": 10_124.0 / 4.0,

    "building_type": "4-story concrete structure with concrete floor",
    "floor_type": "slab",

    "floor_load_capacity_psf": 200.0,
    "floor_load_capacity_kN_m2": 9.58,
    "floor_load_capacity_kg_m2": 9.58 * 1000.0 / 9.81,

    "flood_zone": "A",
    "seismic_design_category": "C",
    "fire_suppression": "double-interlocked pre-action dry pipe",

    "cabinet_density_kVA": 5.0,
    "power_distribution": "120/208V",
    "utility_feeders": 2,

    "ups_configuration": "block redundant",
    "ups_redundancy": "N+1",

    "num_generators": 5,
    "generator_capacity_kW_each": 2500.0,
    "standby_power_redundancy": "N+1",

    "usable_generator_count": 5 - 1,
    "usable_standby_power_kW": (5 - 1) * 2500.0,
    "installed_standby_power_kW": 5 * 2500.0,

    "cooling_configuration": "centrifugal chillers and air handling units",
    "cooling_redundancy": "N+1",

    # Envelope assumptions
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

    # Solar / surface assumptions
    "roof_albedo": 0.35,
    "wall_albedo": 0.30,
    "glass_solar_heat_gain_coeff": 0.35,

    # Datacenter operation
    "assumed_pue": 1.35,
    "heat_rejection_fraction": 0.90,
}


# ============================================================
# CONFIGURATION
# ============================================================

def infer_grid_spacing_from_city_data(city_data):
    """
    Infers dx, dy in meters from transform or bbox_projected.

    mask.shape is assumed to be:
        rows, cols

    bbox width corresponds to columns.
    bbox height corresponds to rows.
    """

    mask = np.asarray(city_data["mask"])
    rows, cols = mask.shape

    transform = city_data.get("transform", None)

    if transform is not None:
        if hasattr(transform, "a") and hasattr(transform, "e"):
            dx = abs(float(transform.a))
            dy = abs(float(transform.e))

            if dx > 0 and dy > 0:
                return dx, dy

        if isinstance(transform, (tuple, list)) and len(transform) >= 6:
            vals = [float(v) for v in transform[:6]]

            possible_dx = abs(vals[1])
            possible_dy = abs(vals[5])

            if possible_dx > 0 and possible_dy > 0:
                return possible_dx, possible_dy

            possible_dx = abs(vals[0])
            possible_dy = abs(vals[4])

            if possible_dx > 0 and possible_dy > 0:
                return possible_dx, possible_dy

    bbox = city_data["bbox_projected"]

    width_m = float(bbox["maxx"] - bbox["minx"])
    height_m = float(bbox["maxy"] - bbox["miny"])

    dx = width_m / max(cols, 1)
    dy = height_m / max(rows, 1)

    return dx, dy


def make_config_from_city_data(
    city_data,
    dt=DEFAULT_DT,
    t_air=DEFAULT_T_AIR,
    humidity=DEFAULT_HUMIDITY,
    solar_radiation=DEFAULT_SOLAR_RADIATION,
):
    mask = np.asarray(city_data["mask"])
    rows, cols = mask.shape

    dx, dy = infer_grid_spacing_from_city_data(city_data)

    if dx <= 0 or dy <= 0:
        raise ValueError(f"Invalid grid spacing: dx={dx}, dy={dy}")

    if dx < 0.5 or dy < 0.5:
        print(
            "Warning: inferred dx/dy are very small. "
            f"dx={dx:.6f}, dy={dy:.6f}. "
            "Check transform/bbox units."
        )

    c_air = DEFAULT_RHO_AIR * DEFAULT_CP_AIR * DEFAULT_MIXING_HEIGHT

    return {
        "rows": rows,
        "cols": cols,
        "dx": dx,
        "dy": dy,
        "dt": dt,

        "t_air": t_air,
        "humidity": humidity,
        "solar_radiation": solar_radiation,

        "rho_air": DEFAULT_RHO_AIR,
        "cp_air": DEFAULT_CP_AIR,
        "mixing_height": DEFAULT_MIXING_HEIGHT,
        "c_air": c_air,

        "alpha_air": DEFAULT_ALPHA_AIR,
        "alpha_building": DEFAULT_ALPHA_BUILDING,
        "alpha_interface": DEFAULT_ALPHA_INTERFACE,

        "base_cooling_rate": DEFAULT_BASE_COOLING_RATE,

        "building_heat_capacity_factor": DEFAULT_BUILDING_HEAT_CAPACITY_FACTOR,

        "load_gamma": DEFAULT_LOAD_GAMMA,
        "max_temp_clip": DEFAULT_MAX_TEMP_CLIP,
    }


# ============================================================
# DATACENTER GEOMETRY AND MATERIAL PROPERTIES
# ============================================================

def estimate_perimeter_from_mask(binary_mask, dx, dy):
    binary_mask = np.asarray(binary_mask).astype(bool)

    padded = np.pad(
        binary_mask,
        pad_width=1,
        mode="constant",
        constant_values=False,
    )

    center = padded[1:-1, 1:-1]
    north = padded[:-2, 1:-1]
    south = padded[2:, 1:-1]
    west = padded[1:-1, :-2]
    east = padded[1:-1, 2:]

    north_edges = (center & ~north).sum()
    south_edges = (center & ~south).sum()
    west_edges = (center & ~west).sum()
    east_edges = (center & ~east).sum()

    perimeter_m = (north_edges + south_edges) * dx + (west_edges + east_edges) * dy

    return float(perimeter_m)


def compute_source_building_geometry(
    central_building_mask,
    dx,
    dy,
    specs=None,
):
    if specs is None:
        specs = DEFAULT_DATACENTER_SPECS

    central = np.asarray(central_building_mask).astype(bool)

    pixel_area_m2 = dx * dy
    footprint_area_m2 = float(central.sum() * pixel_area_m2)
    perimeter_m = estimate_perimeter_from_mask(central, dx, dy)

    num_floors = specs["num_floors"]
    floor_height_m = specs["floor_height_m"]
    total_height_m = num_floors * floor_height_m

    total_wall_area_m2 = perimeter_m * total_height_m

    window_to_wall_ratio = specs["assumed_window_to_wall_ratio"]
    glass_area_m2 = total_wall_area_m2 * window_to_wall_ratio
    opaque_wall_area_m2 = total_wall_area_m2 - glass_area_m2

    return {
        "pixel_area_m2": pixel_area_m2,
        "source_footprint_area_m2": footprint_area_m2,
        "source_perimeter_m": perimeter_m,
        "num_floors": num_floors,
        "floor_height_m": floor_height_m,
        "total_height_m": total_height_m,
        "total_wall_area_m2": total_wall_area_m2,
        "window_to_wall_ratio": window_to_wall_ratio,
        "glass_area_m2": glass_area_m2,
        "opaque_wall_area_m2": opaque_wall_area_m2,
    }


def compute_datacenter_effective_properties_from_mask(
    central_building_mask,
    dx,
    dy,
    specs=None,
):
    if specs is None:
        specs = DEFAULT_DATACENTER_SPECS

    geom = compute_source_building_geometry(
        central_building_mask=central_building_mask,
        dx=dx,
        dy=dy,
        specs=specs,
    )

    colocation_area_m2 = specs["colocation_area_m2"]

    installed_standby_power_W = specs["installed_standby_power_kW"] * 1000.0
    usable_standby_power_W = specs["usable_standby_power_kW"] * 1000.0

    installed_power_density_W_m2 = installed_standby_power_W / colocation_area_m2
    usable_power_density_W_m2 = usable_standby_power_W / colocation_area_m2

    concrete_Cv = specs["concrete_Cv_J_m3K"]
    glass_Cv = specs["glass_Cv_J_m3K"]

    wall_thickness = specs["wall_thickness_m"]
    glass_thickness = specs["glass_thickness_m"]

    k_concrete = specs["k_concrete_W_mK"]
    k_glass = specs["k_glass_W_mK"]

    U_wall = specs["U_wall_W_m2K"]
    U_window = specs["U_window_W_m2K"]

    footprint_area_m2 = max(geom["source_footprint_area_m2"], 1e-6)
    opaque_wall_area_m2 = geom["opaque_wall_area_m2"]
    glass_area_m2 = geom["glass_area_m2"]

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

    alpha_concrete = k_concrete / concrete_Cv
    alpha_glass = k_glass / glass_Cv

    assumed_pue = specs["assumed_pue"]
    heat_rejection_fraction = specs["heat_rejection_fraction"]

    roof_solar_absorption = 1.0 - specs["roof_albedo"]
    wall_solar_absorption = 1.0 - specs["wall_albedo"]
    glass_solar_gain_coeff = specs["glass_solar_heat_gain_coeff"]

    return {
        **geom,

        "colocation_area_m2": colocation_area_m2,

        "installed_standby_power_W": installed_standby_power_W,
        "usable_standby_power_W": usable_standby_power_W,

        "installed_power_density_W_m2": installed_power_density_W_m2,
        "usable_power_density_W_m2": usable_power_density_W_m2,

        "cabinet_density_kVA": specs["cabinet_density_kVA"],

        "C_wall_total_J_K": C_wall_total_J_K,
        "C_glass_total_J_K": C_glass_total_J_K,
        "C_envelope_total_J_K": C_envelope_total_J_K,
        "C_envelope": C_envelope,

        "UA_wall_W_K": UA_wall_W_K,
        "UA_window_W_K": UA_window_W_K,
        "UA_total_W_K": UA_total_W_K,
        "H_envelope_areal_W_m2K": H_envelope_areal_W_m2K,

        "concrete_Cv": concrete_Cv,
        "glass_Cv": glass_Cv,
        "k_concrete": k_concrete,
        "k_glass": k_glass,
        "alpha_concrete": alpha_concrete,
        "alpha_glass": alpha_glass,

        "pue": assumed_pue,
        "heat_rejection_fraction": heat_rejection_fraction,

        "roof_solar_absorption": roof_solar_absorption,
        "wall_solar_absorption": wall_solar_absorption,
        "glass_solar_heat_gain_coeff": glass_solar_gain_coeff,

        "floor_load_capacity_kN_m2": specs["floor_load_capacity_kN_m2"],
        "floor_load_capacity_kg_m2": specs["floor_load_capacity_kg_m2"],
    }


# ============================================================
# CITY DATA EXTRACTION
# ============================================================

def extract_masks_from_city_data(city_data):
    building_mask = np.asarray(city_data["mask"]).astype(bool)
    source_mask_raw = np.asarray(city_data["central_building"]).astype(bool)

    if building_mask.shape != source_mask_raw.shape:
        raise ValueError(
            f"mask and central_building must have same shape. "
            f"Got {building_mask.shape} and {source_mask_raw.shape}."
        )

    if source_mask_raw.sum() == 0:
        raise ValueError(
            "central_building mask is empty. Cannot run datacenter heat simulation."
        )

    building_mask = building_mask | source_mask_raw

    return building_mask, source_mask_raw


# ============================================================
# BASIC MORPHOLOGY AND SMOOTHING
# ============================================================

def smooth_no_wrap(field, smooth_steps=2):
    field = np.asarray(field, dtype=float)

    for _ in range(smooth_steps):
        padded = np.pad(field, pad_width=1, mode="edge")

        center = padded[1:-1, 1:-1]
        north = padded[:-2, 1:-1]
        south = padded[2:, 1:-1]
        west = padded[1:-1, :-2]
        east = padded[1:-1, 2:]

        field = (
            0.55 * center
            + 0.1125 * north
            + 0.1125 * south
            + 0.1125 * west
            + 0.1125 * east
        )

    return field


def dilate_no_wrap(mask, iterations=1):
    mask = np.asarray(mask).astype(bool)
    out = mask.copy()

    for _ in range(iterations):
        padded = np.pad(out.astype(float), pad_width=1, mode="constant")

        dilated = (
            padded[1:-1, 1:-1]
            + padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            + padded[:-2, :-2]
            + padded[:-2, 2:]
            + padded[2:, :-2]
            + padded[2:, 2:]
        ) > 0

        out = dilated

    return out


def make_heat_release_weight_from_central_building(
    central_building_mask,
    release_radius_pixels=1,
    smooth_steps=1,
):
    """
    Datacenter heat starts from the central building mask.

    A small local smoothing is allowed to avoid numerical artifacts, but the
    heat source remains aligned with the central building.
    """

    central = np.asarray(central_building_mask).astype(bool)

    if central.sum() <= 0:
        raise ValueError("central_building_mask is empty.")

    source = central.astype(float)

    if release_radius_pixels > 0:
        local_region = dilate_no_wrap(central, iterations=release_radius_pixels)
    else:
        local_region = central.copy()

    if smooth_steps > 0:
        source = smooth_no_wrap(source, smooth_steps=smooth_steps)
        source = source * local_region.astype(float)

    source = np.maximum(source, 0.0)

    if source.sum() <= 1e-12:
        source = central.astype(float)

    target_sum = float(central.sum())
    source = source * (target_sum / max(source.sum(), 1e-12))

    return source


def make_building_interface_field(building_mask, smooth_steps=3):
    """
    High near building-air interfaces.

    Represents solid-air heat exchange through walls and windows.
    """

    building = np.asarray(building_mask).astype(float)

    padded = np.pad(building, pad_width=1, mode="edge")

    north = padded[:-2, 1:-1]
    south = padded[2:, 1:-1]
    west = padded[1:-1, :-2]
    east = padded[1:-1, 2:]

    neighbor_mean = 0.25 * (north + south + west + east)

    interface = np.abs(building - neighbor_mean)
    interface = smooth_no_wrap(interface, smooth_steps=smooth_steps)

    max_val = interface.max()

    if max_val > 1e-12:
        interface = interface / max_val

    return interface


def make_solar_absorption_field(
    building_mask,
    central_building_mask,
    dc_props,
):
    """
    Creates a solar absorption field.

    Solar radiation can heat:
    - open ground / streets;
    - passive building roofs;
    - central datacenter roof.
    """

    building = np.asarray(building_mask).astype(bool)
    central = np.asarray(central_building_mask).astype(bool)

    solar_absorption = np.zeros_like(building, dtype=float)

    open_ground_absorption = 0.75
    passive_building_roof_absorption = 0.65
    datacenter_roof_absorption = dc_props["roof_solar_absorption"]

    solar_absorption[~building] = open_ground_absorption
    solar_absorption[building] = passive_building_roof_absorption
    solar_absorption[central] = datacenter_roof_absorption

    return solar_absorption


# ============================================================
# NUMERICAL OPERATORS
# ============================================================

def upwind_gradient_no_wrap(T, wind_x, wind_y, dx, dy):
    """
    First-order upwind gradient for one constant wind vector.
    """

    T = np.asarray(T)

    dTdx_back = np.zeros_like(T)
    dTdx_forw = np.zeros_like(T)

    dTdy_back = np.zeros_like(T)
    dTdy_forw = np.zeros_like(T)

    dTdx_back[1:, :] = (T[1:, :] - T[:-1, :]) / dx
    dTdx_back[0, :] = dTdx_back[1, :]

    dTdx_forw[:-1, :] = (T[1:, :] - T[:-1, :]) / dx
    dTdx_forw[-1, :] = dTdx_forw[-2, :]

    dTdy_back[:, 1:] = (T[:, 1:] - T[:, :-1]) / dy
    dTdy_back[:, 0] = dTdy_back[:, 1]

    dTdy_forw[:, :-1] = (T[:, 1:] - T[:, :-1]) / dy
    dTdy_forw[:, -1] = dTdy_forw[:, -2]

    dTdx = dTdx_back if wind_x >= 0 else dTdx_forw
    dTdy = dTdy_back if wind_y >= 0 else dTdy_forw

    return dTdx, dTdy


def variable_diffusion_no_wrap(T, alpha, dx, dy):
    """
    Computes div(alpha grad T) with no periodic wraparound.
    """

    T_pad = np.pad(T, pad_width=1, mode="edge")
    a_pad = np.pad(alpha, pad_width=1, mode="edge")

    T_c = T_pad[1:-1, 1:-1]
    T_n = T_pad[:-2, 1:-1]
    T_s = T_pad[2:, 1:-1]
    T_w = T_pad[1:-1, :-2]
    T_e = T_pad[1:-1, 2:]

    a_c = a_pad[1:-1, 1:-1]

    a_n = 0.5 * (a_c + a_pad[:-2, 1:-1])
    a_s = 0.5 * (a_c + a_pad[2:, 1:-1])
    a_w = 0.5 * (a_c + a_pad[1:-1, :-2])
    a_e = 0.5 * (a_c + a_pad[1:-1, 2:])

    diff_x = (a_s * (T_s - T_c) - a_n * (T_c - T_n)) / dx**2
    diff_y = (a_e * (T_e - T_c) - a_w * (T_c - T_w)) / dy**2

    return diff_x + diff_y


def apply_open_boundary(T, t_air):
    T[0, :] = 0.5 * T[1, :] + 0.5 * t_air
    T[-1, :] = 0.5 * T[-2, :] + 0.5 * t_air
    T[:, 0] = 0.5 * T[:, 1] + 0.5 * t_air
    T[:, -1] = 0.5 * T[:, -2] + 0.5 * t_air

    return T


def compute_stable_substep_dt(
    dt_requested,
    dx,
    dy,
    alpha_max,
    wind_x,
    wind_y,
    safety=0.30,
):
    eps = 1e-12

    diffusion_limit = 1.0 / (
        2.0 * max(alpha_max, eps) * (1.0 / dx**2 + 1.0 / dy**2)
    )

    advection_limit_x = dx / max(abs(wind_x), eps)
    advection_limit_y = dy / max(abs(wind_y), eps)

    stable_dt = safety * min(diffusion_limit, advection_limit_x, advection_limit_y)
    stable_dt = min(dt_requested, stable_dt)

    n_substeps = int(np.ceil(dt_requested / max(stable_dt, eps)))
    n_substeps = max(n_substeps, 1)

    dt_sub = dt_requested / n_substeps

    return dt_sub, n_substeps


# ============================================================
# MAIN PHYSICAL SIMULATION
# ============================================================

def simulate_city_heat_from_city_data(
    city_data,
    load,
    wind_x,
    wind_y,
    t_air=DEFAULT_T_AIR,
    humidity=DEFAULT_HUMIDITY,
    solar_radiation=DEFAULT_SOLAR_RADIATION,
    datacenter_specs=None,
    dt=DEFAULT_DT,
    steps=DEFAULT_STEPS,
    save_every=DEFAULT_SAVE_EVERY,
    release_radius_pixels=1,
    release_smooth_steps=1,
    include_solar=True,
    return_states=False,
    max_temp_clip=DEFAULT_MAX_TEMP_CLIP,
    verbose=True,
):
    """
    Real-city heat plume model.

    Wind:
        one constant vector, wind_x and wind_y.

    Buildings:
        do not change the wind;
        do not generate datacenter load heat;
        affect thermal transport, heat capacity, wall/window dissipation,
        and solar absorption.

    Heat source:
        datacenter load heat starts from city_data["central_building"].
    """

    if datacenter_specs is None:
        datacenter_specs = DEFAULT_DATACENTER_SPECS

    config = make_config_from_city_data(
        city_data=city_data,
        dt=dt,
        t_air=t_air,
        humidity=humidity,
        solar_radiation=solar_radiation,
    )

    rows = config["rows"]
    cols = config["cols"]
    dx = config["dx"]
    dy = config["dy"]

    c_air = config["c_air"]

    alpha_air = config["alpha_air"]
    alpha_building = config["alpha_building"]
    alpha_interface = config["alpha_interface"]

    base_cooling_rate = config["base_cooling_rate"]
    building_heat_capacity_factor = config["building_heat_capacity_factor"]
    load_gamma = config["load_gamma"]

    building_mask, central_building_mask = extract_masks_from_city_data(city_data)

    dc_props = compute_datacenter_effective_properties_from_mask(
        central_building_mask=central_building_mask,
        dx=dx,
        dy=dy,
        specs=datacenter_specs,
    )

    T = np.ones((rows, cols), dtype=float) * t_air

    # --------------------------------------------------------
    # Datacenter heat source aligned to central building mask
    # --------------------------------------------------------

    release_weight = make_heat_release_weight_from_central_building(
        central_building_mask=central_building_mask,
        release_radius_pixels=release_radius_pixels,
        smooth_steps=release_smooth_steps,
    )

    # --------------------------------------------------------
    # Material-dependent thermal transport
    # --------------------------------------------------------

    interface_field = make_building_interface_field(
        building_mask=building_mask,
        smooth_steps=3,
    )

    alpha = np.where(
        building_mask,
        alpha_building,
        alpha_air,
    )

    alpha = (
        alpha * (1.0 - interface_field)
        + alpha_interface * interface_field
    )

    # Slight local physical/numerical mixing around source boundary,
    # but without shifting the origin.
    release_mix = smooth_no_wrap(release_weight, smooth_steps=2)

    if release_mix.max() > 1e-12:
        release_mix = release_mix / release_mix.max()

    alpha = alpha + 3.0 * release_mix

    # --------------------------------------------------------
    # Effective heat capacity
    # --------------------------------------------------------

    C_eff = np.where(
        building_mask,
        c_air * building_heat_capacity_factor,
        c_air,
    )

    central_float = central_building_mask.astype(float)
    C_eff = C_eff + central_float * dc_props["C_envelope"]

    # --------------------------------------------------------
    # Weather cooling
    # --------------------------------------------------------

    wind_speed = float(np.sqrt(wind_x**2 + wind_y**2))

    wind_cooling_factor = 1.0 + 0.18 * wind_speed
    humidity_cooling_factor = 1.0 - 0.35 * humidity

    cooling_rate = base_cooling_rate * wind_cooling_factor * humidity_cooling_factor

    # --------------------------------------------------------
    # Datacenter load heat
    # --------------------------------------------------------

    nonlinear_load = load ** load_gamma

    Q_power_density = dc_props["usable_power_density_W_m2"] * nonlinear_load
    Q_facility_density = Q_power_density * dc_props["pue"]
    Q_waste_density = Q_facility_density * dc_props["heat_rejection_fraction"]

    Q_datacenter = Q_waste_density * release_weight

    # --------------------------------------------------------
    # Solar radiation
    # --------------------------------------------------------

    solar_absorption = make_solar_absorption_field(
        building_mask=building_mask,
        central_building_mask=central_building_mask,
        dc_props=dc_props,
    )

    if include_solar:
        Q_solar = solar_radiation * solar_absorption
    else:
        Q_solar = np.zeros_like(T)

    footprint_area = max(dc_props["source_footprint_area_m2"], 1e-6)

    Q_wall_solar_total_W = (
        solar_radiation
        * dc_props["wall_solar_absorption"]
        * dc_props["opaque_wall_area_m2"]
        * 0.25
    )

    Q_glass_solar_total_W = (
        solar_radiation
        * dc_props["glass_solar_heat_gain_coeff"]
        * dc_props["glass_area_m2"]
        * 0.25
    )

    Q_wall_window_solar_density = (
        Q_wall_solar_total_W + Q_glass_solar_total_W
    ) / footprint_area

    if include_solar:
        Q_solar = Q_solar + central_float * Q_wall_window_solar_density

    # --------------------------------------------------------
    # Wall/window dissipation
    # --------------------------------------------------------

    H_env = dc_props["H_envelope_areal_W_m2K"]

    envelope_exchange_factor = (1.0 + 0.20 * wind_speed) * (1.0 - 0.15 * humidity)
    H_env_effective = H_env * envelope_exchange_factor

    # --------------------------------------------------------
    # Stability
    # --------------------------------------------------------

    alpha_max = float(np.max(alpha))

    dt_sub, n_substeps = compute_stable_substep_dt(
        dt_requested=dt,
        dx=dx,
        dy=dy,
        alpha_max=alpha_max,
        wind_x=wind_x,
        wind_y=wind_y,
        safety=0.30,
    )

    if verbose:
        print("\nSimulation grid:")
        print(f"shape: {rows} x {cols}")
        print(f"dx = {dx:.3f} m, dy = {dy:.3f} m")
        print(f"wind vector = ({wind_x:.3f}, {wind_y:.3f}) m/s")
        print(f"solar radiation = {solar_radiation:.2f} W/m2")
        print(f"requested dt = {dt:.4f} s")
        print(f"stable substep dt = {dt_sub:.6f} s")
        print(f"substeps per step = {n_substeps}")
        print(f"source footprint area = {dc_props['source_footprint_area_m2']:.2f} m2")
        print(f"source heat density = {Q_waste_density:.2f} W/m2")
        print(f"H envelope base = {H_env:.4f} W/(m2 K)")
        print(f"H envelope effective = {H_env_effective:.4f} W/(m2 K)")
        print(f"alpha min/max = {alpha.min():.3f} / {alpha.max():.3f}")
        print(f"release_weight sum = {release_weight.sum():.6f}")
        print(f"release_weight min/max = {release_weight.min():.6f} / {release_weight.max():.6f}")

    states = []

    for n in range(steps):
        for _ in range(n_substeps):
            dTdx, dTdy = upwind_gradient_no_wrap(
                T=T,
                wind_x=wind_x,
                wind_y=wind_y,
                dx=dx,
                dy=dy,
            )

            dTdt_adv = -(wind_x * dTdx + wind_y * dTdy)

            dTdt_diff = variable_diffusion_no_wrap(
                T=T,
                alpha=alpha,
                dx=dx,
                dy=dy,
            )

            dTdt_datacenter = Q_datacenter / C_eff

            dTdt_solar = Q_solar / C_eff

            dTdt_cool = -cooling_rate * (T - t_air)

            dTdt_envelope_loss = (
                -central_float * H_env_effective * (T - t_air) / C_eff
            )

            excess = np.maximum(T - t_air, 0.0)
            dTdt_hot_cooling = -2.5e-6 * excess**2

            dTdt = (
                dTdt_adv
                + dTdt_diff
                + dTdt_datacenter
                + dTdt_solar
                + dTdt_cool
                + dTdt_envelope_loss
                + dTdt_hot_cooling
            )

            T = T + dt_sub * dTdt
            T = apply_open_boundary(T, t_air)
            T = np.clip(T, t_air - 2.0, max_temp_clip)

        if return_states and n % save_every == 0:
            states.append(T.copy())

    anomaly = T - t_air

    metrics = {
        "location": city_data.get("location", "unknown"),
        "load": float(load),

        "rows": int(rows),
        "cols": int(cols),
        "dx": float(dx),
        "dy": float(dy),

        "wind_x": float(wind_x),
        "wind_y": float(wind_y),
        "wind_speed": float(wind_speed),

        "t_air": float(t_air),
        "humidity": float(humidity),
        "solar_radiation": float(solar_radiation),
        "include_solar": bool(include_solar),

        "dt_requested": float(dt),
        "dt_sub": float(dt_sub),
        "n_substeps": int(n_substeps),

        "central_building_found": bool(city_data.get("central_building_found", True)),
        "central_building_method": city_data.get("central_building_method", None),

        "max_temp": float(T.max()),
        "mean_temp": float(T.mean()),
        "max_anomaly": float(anomaly.max()),
        "mean_anomaly": float(anomaly.mean()),
        "total_anomaly": float(anomaly.sum()),

        "hot_area_1C_pixels": int(np.sum(anomaly > 1.0)),
        "hot_area_2C_pixels": int(np.sum(anomaly > 2.0)),
        "hot_area_5C_pixels": int(np.sum(anomaly > 5.0)),

        "hot_area_1C_m2": float(np.sum(anomaly > 1.0) * dx * dy),
        "hot_area_2C_m2": float(np.sum(anomaly > 2.0) * dx * dy),
        "hot_area_5C_m2": float(np.sum(anomaly > 5.0) * dx * dy),

        "source_footprint_area_m2": float(dc_props["source_footprint_area_m2"]),
        "source_perimeter_m": float(dc_props["source_perimeter_m"]),

        "usable_power_density_W_m2": float(dc_props["usable_power_density_W_m2"]),
        "source_heat_density_W_m2": float(Q_waste_density),

        "window_to_wall_ratio": float(dc_props["window_to_wall_ratio"]),
        "glass_area_m2": float(dc_props["glass_area_m2"]),
        "opaque_wall_area_m2": float(dc_props["opaque_wall_area_m2"]),
        "total_wall_area_m2": float(dc_props["total_wall_area_m2"]),
        "UA_total_W_K": float(dc_props["UA_total_W_K"]),
        "H_envelope_areal_W_m2K": float(dc_props["H_envelope_areal_W_m2K"]),
        "H_envelope_effective_W_m2K": float(H_env_effective),

        "Q_wall_window_solar_density_W_m2": float(Q_wall_window_solar_density),
        "Q_solar_mean_W_m2": float(Q_solar.mean()),
        "Q_solar_max_W_m2": float(Q_solar.max()),

        "alpha_min": float(alpha.min()),
        "alpha_max": float(alpha.max()),

        "release_weight_sum": float(release_weight.sum()),
        "release_weight_min": float(release_weight.min()),
        "release_weight_max": float(release_weight.max()),

        "bbox_latlon": city_data.get("bbox_latlon", None),
        "bbox_projected": city_data.get("bbox_projected", None),
        "crs_projected": city_data.get("crs_projected", None),
    }

    result = {
        "metrics": metrics,
        "temperature": T,
        "anomaly": anomaly,
        "building_mask": building_mask,
        "central_building_mask": central_building_mask,
        "release_weight": release_weight,
        "source_weight": release_weight,
        "interface_field": interface_field,
        "alpha": alpha,
        "solar_absorption": solar_absorption,
        "Q_solar": Q_solar,
        "Q_datacenter": Q_datacenter,
        "config": config,
        "dc_props": dc_props,
        "city_data": city_data,
    }

    if return_states:
        result["states"] = np.array(states)

    return result


# ============================================================
# PLOTTING
# ============================================================

def _safe_vmin_vmax(arr, default_pad=1e-6):
    arr = np.asarray(arr)
    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        return 0.0, 1.0

    vmin = float(finite.min())
    vmax = float(finite.max())

    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.01, default_pad)
        vmin -= pad
        vmax += pad

    return vmin, vmax


def plot_real_city_masks(result):
    building_mask = result["building_mask"]
    central = result["central_building_mask"]
    location = result["metrics"]["location"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(building_mask, origin="upper", cmap="gray_r")
    axes[0].set_title(f"{location}\nall buildings")
    axes[0].set_xlabel("col")
    axes[0].set_ylabel("row")

    axes[1].imshow(building_mask, origin="upper", cmap="gray_r")
    axes[1].contour(central, levels=[0.5], colors="red", linewidths=1.5)
    axes[1].set_title("central datacenter footprint")
    axes[1].set_xlabel("col")
    axes[1].set_ylabel("row")

    plt.tight_layout()
    plt.show()


def plot_release_weight(result):
    release_weight = np.asarray(result["release_weight"], dtype=float)
    central = result["central_building_mask"]

    vmin, vmax = _safe_vmin_vmax(release_weight)

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(
        release_weight,
        origin="upper",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )

    if np.any(central):
        ax.contour(central, levels=[0.5], colors="white", linewidths=1.5)

    fig.colorbar(im, ax=ax, label="heat release weight")
    ax.set_title("Datacenter heat source aligned with central mask")
    ax.set_xlabel("col")
    ax.set_ylabel("row")

    plt.tight_layout()
    plt.show()


def plot_material_fields(result):
    alpha = result["alpha"]
    interface = result["interface_field"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(alpha, origin="upper", cmap="viridis")
    axes[0].set_title("material-dependent heat diffusivity")
    axes[0].set_xlabel("col")
    axes[0].set_ylabel("row")
    fig.colorbar(im0, ax=axes[0], label="m²/s")

    im1 = axes[1].imshow(interface, origin="upper", cmap="magma")
    axes[1].set_title("building-air interface field")
    axes[1].set_xlabel("col")
    axes[1].set_ylabel("row")
    fig.colorbar(im1, ax=axes[1], label="interface strength")

    plt.tight_layout()
    plt.show()


def plot_solar_fields(result):
    solar_absorption = result["solar_absorption"]
    Q_solar = result["Q_solar"]
    central = result["central_building_mask"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(solar_absorption, origin="upper", cmap="viridis")
    axes[0].contour(central, levels=[0.5], colors="white", linewidths=1.4)
    axes[0].set_title("solar absorption field")
    axes[0].set_xlabel("col")
    axes[0].set_ylabel("row")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(Q_solar, origin="upper", cmap="inferno")
    axes[1].contour(central, levels=[0.5], colors="white", linewidths=1.4)
    axes[1].set_title("solar heat flux [W/m²]")
    axes[1].set_xlabel("col")
    axes[1].set_ylabel("row")
    fig.colorbar(im1, ax=axes[1], label="W/m²")

    plt.tight_layout()
    plt.show()


def plot_heat_terms(result):
    Q_datacenter = result["Q_datacenter"]
    Q_solar = result["Q_solar"]
    central = result["central_building_mask"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(Q_datacenter, origin="upper", cmap="inferno")
    axes[0].contour(central, levels=[0.5], colors="white", linewidths=1.4)
    axes[0].set_title("datacenter load heat [W/m²]")
    axes[0].set_xlabel("col")
    axes[0].set_ylabel("row")
    fig.colorbar(im0, ax=axes[0], label="W/m²")

    im1 = axes[1].imshow(Q_solar, origin="upper", cmap="inferno")
    axes[1].contour(central, levels=[0.5], colors="white", linewidths=1.4)
    axes[1].set_title("solar heat [W/m²]")
    axes[1].set_xlabel("col")
    axes[1].set_ylabel("row")
    fig.colorbar(im1, ax=axes[1], label="W/m²")

    plt.tight_layout()
    plt.show()


def plot_real_city_temperature(result):
    T = result["temperature"]
    anomaly = result["anomaly"]
    building_mask = result["building_mask"]
    central = result["central_building_mask"]
    t_air = result["config"]["t_air"]
    location = result["metrics"]["location"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    t_vmin = t_air
    t_vmax = max(t_air + 0.5, float(np.nanpercentile(T, 99)))

    if np.isclose(t_vmin, t_vmax):
        t_vmax = t_vmin + 1.0

    im0 = axes[0].imshow(
        T,
        origin="upper",
        cmap="inferno",
        vmin=t_vmin,
        vmax=t_vmax,
    )

    axes[0].contour(building_mask, levels=[0.5], colors="cyan", linewidths=0.15)
    axes[0].contour(central, levels=[0.5], colors="white", linewidths=1.4)
    axes[0].set_title(f"{location}\nfinal temperature [°C]")
    axes[0].set_xlabel("col")
    axes[0].set_ylabel("row")
    fig.colorbar(im0, ax=axes[0], label="Temperature [°C]")

    a_vmin = 0.0
    a_vmax = max(0.2, float(np.nanpercentile(anomaly, 99)))

    if np.isclose(a_vmin, a_vmax):
        a_vmax = a_vmin + 1.0

    im1 = axes[1].imshow(
        anomaly,
        origin="upper",
        cmap="coolwarm",
        vmin=a_vmin,
        vmax=a_vmax,
    )

    axes[1].contour(building_mask, levels=[0.5], colors="black", linewidths=0.15)
    axes[1].contour(central, levels=[0.5], colors="white", linewidths=1.4)
    axes[1].set_title("temperature anomaly")
    axes[1].set_xlabel("col")
    axes[1].set_ylabel("row")
    fig.colorbar(im1, ax=axes[1], label="Temperature anomaly [°C]")

    plt.tight_layout()
    plt.show()


def animate_real_city_result(result, wind_x, wind_y):
    if "states" not in result:
        raise ValueError("No states found. Run with return_states=True.")

    states = result["states"]
    building_mask = result["building_mask"]
    central = result["central_building_mask"]
    location = result["metrics"]["location"]
    t_air = result["config"]["t_air"]

    fig, ax = plt.subplots(figsize=(7, 6))

    vmin = t_air
    vmax = max(t_air + 0.5, float(np.nanpercentile(states, 99)))

    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0

    im = ax.imshow(
        states[0],
        origin="upper",
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )

    ax.contour(building_mask, levels=[0.5], colors="cyan", linewidths=0.15)
    ax.contour(central, levels=[0.5], colors="white", linewidths=1.2)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Temperature [°C]")

    title = ax.set_title(f"{location}: heat plume")
    ax.set_xlabel("col")
    ax.set_ylabel("row")

    arrow_scale = 20
    ax.arrow(
        8,
        8,
        arrow_scale * wind_x,
        arrow_scale * wind_y,
        color="white",
        width=0.8,
        head_width=4,
        length_includes_head=True,
    )
    ax.text(8, 17, "wind", color="white", fontsize=11)

    def update(frame):
        im.set_data(states[frame])
        title.set_text(f"{location}: heat plume, frame = {frame}")
        return [im, title]

    ani = FuncAnimation(
        fig,
        update,
        frames=len(states),
        interval=50,
        blit=False,
    )

    plt.show()
    return ani


# ============================================================
# EXAMPLE / MAIN
# ============================================================

def run_example_with_latlon(
    lat,
    lon,
    n_meters=400,
    nx=160,
    ny=160,
    load=0.45,
    wind_x=0.75,
    wind_y=0.35,
    t_air=26.0,
    humidity=0.60,
    solar_radiation=650.0,
    include_solar=True,
    steps=900,
    make_plots=True,
    verbose=True,
):
    """
    Uses latlon_to_central_building_mask from buildings_latlon.py.

    Returns:
        result dictionary with temperature, anomaly, masks, terms, and metrics.
    """

    city_data = latlon_to_central_building_mask(
        lat=lat,
        lon=lon,
        n_meters=n_meters,
        Nx=nx,
        Ny=ny,
    )

    if not city_data["central_building_found"]:
        print("Warning: central building was not confidently found.")
        print("Method:", city_data.get("central_building_method"))

    result = simulate_city_heat_from_city_data(
        city_data=city_data,
        load=load,
        wind_x=wind_x,
        wind_y=wind_y,
        t_air=t_air,
        humidity=humidity,
        solar_radiation=solar_radiation,
        datacenter_specs=DEFAULT_DATACENTER_SPECS,
        dt=1.0,
        steps=steps,
        save_every=20,
        release_radius_pixels=1,
        release_smooth_steps=1,
        include_solar=include_solar,
        return_states=True,
        max_temp_clip=55.0,
        verbose=verbose,
    )

    if make_plots:
        plot_real_city_masks(result)
        plot_release_weight(result)
        plot_material_fields(result)
        plot_solar_fields(result)
        plot_heat_terms(result)
        plot_real_city_temperature(result)
        animate_real_city_result(result, wind_x=wind_x, wind_y=wind_y)

    return result


def main(
    lat=None,
    lon=None,
    n_meters=400,
    nx=160,
    ny=160,
    load=0.45,
    wind_x=0.75,
    wind_y=0.35,
    t_air=26.0,
    humidity=0.60,
    solar_radiation=650.0,
    include_solar=True,
    steps=900,
    make_plots=True,
    verbose=True,
):
    """
    Main function.

    It directly uses:
        from buildings_latlon import latlon_to_central_building_mask

    Example:

        result = main(
            lat=43.6532,
            lon=-79.3832,
            n_meters=400,
            nx=160,
            ny=160,
        )
    """

    if lat is None or lon is None:
        return {
            "message": (
                "Provide lat and lon to run. Example: "
                "result = main(lat=43.6532, lon=-79.3832)"
            ),
            "result": None,
        }

    result = run_example_with_latlon(
        lat=lat,
        lon=lon,
        n_meters=n_meters,
        nx=nx,
        ny=ny,
        load=load,
        wind_x=wind_x,
        wind_y=wind_y,
        t_air=t_air,
        humidity=humidity,
        solar_radiation=solar_radiation,
        include_solar=include_solar,
        steps=steps,
        make_plots=make_plots,
        verbose=verbose,
    )

    return result


if __name__ == "__main__":
    output = main(
        lat=43.6532,
        lon=-79.3832,
        n_meters=400,
        nx=160,
        ny=160,
        make_plots=True,
        verbose=True,
    )