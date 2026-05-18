import numpy as np
import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt

from shapely.geometry import box
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from geopy.geocoders import Nominatim


def latlon_to_central_building_mask(
    lat: float,
    lon: float,
    n_meters: float,
    Nx: int,
    Ny: int,
    reverse_geocode: bool = True,
    all_touched: bool = True,
    use_nearest_if_not_inside: bool = True,
):
    """
    Create two binary masks around a latitude/longitude point:

        1. mask:
            All buildings in the selected area.

        2. central_building:
            Only the building that contains the input point.
            If the point is not inside any building, the nearest building is used
            when use_nearest_if_not_inside=True.

    Both masks have shape (Ny, Nx), and both use the same raster transform.

    Parameters
    ----------
    lat : float
        Latitude of the center point.

    lon : float
        Longitude of the center point.

    n_meters : float
        Search radius in meters around the center point.

    Nx : int
        Number of grid cells in the x-direction.

    Ny : int
        Number of grid cells in the y-direction.

    reverse_geocode : bool
        If True, returns a human-readable address.

    all_touched : bool
        If True, every pixel touched by a building polygon is marked as 1.

    use_nearest_if_not_inside : bool
        If True, uses the nearest building if the point is not inside a building.

    Returns
    -------
    result : dict
        {
            "location": str or None,
            "mask": np.ndarray,                  # all buildings mask, shape (Ny, Nx)
            "central_building": np.ndarray,      # central building mask, shape (Ny, Nx)
            "central_building_polygon": GeoDataFrame,
            "all_buildings": GeoDataFrame,
            "bbox_latlon": dict,
            "bbox_projected": dict,
            "transform": rasterio transform,
            "crs_projected": CRS,
            "central_building_found": bool,
            "central_building_method": str or None,
        }
    """

    # ---------------------------------------------------------
    # 1. Optional reverse geocoding
    # ---------------------------------------------------------
    location_name = None

    if reverse_geocode:
        try:
            geolocator = Nominatim(user_agent="central-building-mask-generator")
            location = geolocator.reverse((lat, lon), language="en", timeout=10)

            if location is not None:
                location_name = location.address

        except Exception:
            location_name = None

    # ---------------------------------------------------------
    # 2. Create center point in EPSG:4326
    # ---------------------------------------------------------
    center_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([lon], [lat]),
        crs="EPSG:4326",
    )

    # ---------------------------------------------------------
    # 3. Project center to a local metric CRS
    # ---------------------------------------------------------
    center_projected = ox.projection.project_gdf(center_gdf)
    projected_crs = center_projected.crs

    center_point_projected = center_projected.geometry.iloc[0]

    cx = center_point_projected.x
    cy = center_point_projected.y

    # ---------------------------------------------------------
    # 4. Create metric bounding box
    # ---------------------------------------------------------
    minx = cx - n_meters
    maxx = cx + n_meters
    miny = cy - n_meters
    maxy = cy + n_meters

    bbox_projected_geom = box(minx, miny, maxx, maxy)

    bbox_projected_gdf = gpd.GeoDataFrame(
        geometry=[bbox_projected_geom],
        crs=projected_crs,
    )

    # Convert bbox to lat/lon for OSMnx
    bbox_latlon_gdf = bbox_projected_gdf.to_crs("EPSG:4326")
    west, south, east, north = bbox_latlon_gdf.total_bounds

    # OSMnx expects bbox as: left, bottom, right, top
    # That is: west, south, east, north
    bbox = (west, south, east, north)

    # ---------------------------------------------------------
    # 5. Download building footprints from OpenStreetMap
    # ---------------------------------------------------------
    ox.settings.use_cache = True
    ox.settings.log_console = False

    tags = {"building": True}

    try:
        buildings = ox.features_from_bbox(
            bbox=bbox,
            tags=tags,
        )

    except TypeError:
        # Fallback for older OSMnx versions
        try:
            buildings = ox.geometries_from_bbox(
                north=north,
                south=south,
                east=east,
                west=west,
                tags=tags,
            )
        except Exception:
            buildings = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    except Exception:
        buildings = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # ---------------------------------------------------------
    # 6. Clean downloaded building geometries
    # ---------------------------------------------------------
    if buildings is None or len(buildings) == 0:
        all_buildings_projected = gpd.GeoDataFrame(
            geometry=[],
            crs=projected_crs,
        )

    else:
        buildings = buildings.reset_index()

        if "geometry" not in buildings.columns:
            all_buildings_projected = gpd.GeoDataFrame(
                geometry=[],
                crs=projected_crs,
            )

        else:
            buildings = gpd.GeoDataFrame(
                buildings,
                geometry="geometry",
                crs="EPSG:4326",
            )

            buildings = buildings[
                buildings.geometry.notnull()
                & ~buildings.geometry.is_empty
                & buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
            ].copy()

            if len(buildings) == 0:
                all_buildings_projected = gpd.GeoDataFrame(
                    geometry=[],
                    crs=projected_crs,
                )

            else:
                all_buildings_projected = buildings.to_crs(projected_crs)

                # Fix invalid geometries
                all_buildings_projected["geometry"] = (
                    all_buildings_projected.geometry.buffer(0)
                )

                all_buildings_projected = all_buildings_projected[
                    all_buildings_projected.geometry.notnull()
                    & ~all_buildings_projected.geometry.is_empty
                ].copy()

                # Clip to requested metric area
                all_buildings_projected = gpd.clip(
                    all_buildings_projected,
                    bbox_projected_gdf,
                )

                all_buildings_projected = all_buildings_projected[
                    all_buildings_projected.geometry.notnull()
                    & ~all_buildings_projected.geometry.is_empty
                ].copy()

    # ---------------------------------------------------------
    # 7. Select central building polygon
    # ---------------------------------------------------------
    central_building_found = False
    central_building_method = None

    if len(all_buildings_projected) == 0:
        central_building_polygon = gpd.GeoDataFrame(
            geometry=[],
            crs=projected_crs,
        )

    else:
        # First try: building that contains or touches the input coordinate
        contains_center = all_buildings_projected[
            all_buildings_projected.geometry.contains(center_point_projected)
            | all_buildings_projected.geometry.touches(center_point_projected)
        ].copy()

        if len(contains_center) > 0:
            central_building_polygon = contains_center.iloc[[0]].copy()
            central_building_found = True
            central_building_method = "contains_center"

        elif use_nearest_if_not_inside:
            # Second try: nearest building to the input coordinate
            distances = all_buildings_projected.geometry.distance(center_point_projected)
            nearest_idx = distances.idxmin()

            central_building_polygon = all_buildings_projected.loc[[nearest_idx]].copy()
            central_building_found = True
            central_building_method = "nearest_to_center"

        else:
            central_building_polygon = gpd.GeoDataFrame(
                geometry=[],
                crs=projected_crs,
            )

    # ---------------------------------------------------------
    # 8. Raster transform for both masks
    # ---------------------------------------------------------
    transform = from_bounds(
        west=minx,
        south=miny,
        east=maxx,
        north=maxy,
        width=Nx,
        height=Ny,
    )

    # ---------------------------------------------------------
    # 9. Rasterize all buildings mask
    # ---------------------------------------------------------
    if len(all_buildings_projected) == 0:
        all_buildings_mask = np.zeros((Ny, Nx), dtype=np.uint8)

    else:
        all_shapes = [
            (geom, 1)
            for geom in all_buildings_projected.geometry
            if geom is not None and not geom.is_empty
        ]

        all_buildings_mask = rasterize(
            shapes=all_shapes,
            out_shape=(Ny, Nx),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=all_touched,
        )

    # ---------------------------------------------------------
    # 10. Rasterize central building mask
    # ---------------------------------------------------------
    if len(central_building_polygon) == 0:
        central_building_mask = np.zeros((Ny, Nx), dtype=np.uint8)

    else:
        central_shapes = [
            (geom, 1)
            for geom in central_building_polygon.geometry
            if geom is not None and not geom.is_empty
        ]

        central_building_mask = rasterize(
            shapes=central_shapes,
            out_shape=(Ny, Nx),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=all_touched,
        )

    # Make sure the central building mask never invents pixels
    # outside the all-buildings mask.
    central_building_mask = central_building_mask * all_buildings_mask

    return {
        "location": location_name,

        # All buildings mask
        "mask": all_buildings_mask,

        # Central building mask, same size and same pixel space as mask
        "central_building": central_building_mask,

        # Polygons for debugging / plotting
        "central_building_polygon": central_building_polygon,
        "all_buildings": all_buildings_projected,

        "bbox_latlon": {
            "west": west,
            "south": south,
            "east": east,
            "north": north,
        },
        "bbox_projected": {
            "minx": minx,
            "miny": miny,
            "maxx": maxx,
            "maxy": maxy,
        },
        "transform": transform,
        "crs_projected": projected_crs,
        "central_building_found": central_building_found,
        "central_building_method": central_building_method,
    }


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":

    # Example: University of Waterloo area
    lat = 43.4723
    lon = -80.5449

    n_meters = 500
    Nx = 256
    Ny = 256

    result = latlon_to_central_building_mask(
        lat=lat,
        lon=lon,
        n_meters=n_meters,
        Nx=Nx,
        Ny=Ny,
        reverse_geocode=True,
        all_touched=True,
        use_nearest_if_not_inside=True,
    )

    # ---------------------------------------------------------
    # Extract results
    # ---------------------------------------------------------
    mask = result["mask"]
    central_building = result["central_building"]

    all_buildings = result["all_buildings"]
    central_building_polygon = result["central_building_polygon"]

    print("Location:")
    print(result["location"])
    print()

    print("Projected CRS:", result["crs_projected"])
    print("Number of downloaded buildings:", len(all_buildings))
    print("Central building found:", result["central_building_found"])
    print("Central building method:", result["central_building_method"])
    print("All buildings mask shape:", mask.shape)
    print("Central building mask shape:", central_building.shape)
    print("All buildings mask sum:", int(mask.sum()))
    print("Central building mask sum:", int(central_building.sum()))
    print("Unique values in all-building mask:", np.unique(mask))
    print("Unique values in central-building mask:", np.unique(central_building))

    # ---------------------------------------------------------
    # Get projected bounding box
    # ---------------------------------------------------------
    minx = result["bbox_projected"]["minx"]
    miny = result["bbox_projected"]["miny"]
    maxx = result["bbox_projected"]["maxx"]
    maxy = result["bbox_projected"]["maxy"]

    bbox_projected = gpd.GeoDataFrame(
        geometry=[box(minx, miny, maxx, maxy)],
        crs=result["crs_projected"],
    )

    # ---------------------------------------------------------
    # Plot 1: raster masks
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(mask, origin="upper", cmap="gray")
    axes[0].set_title("All Buildings Mask")
    axes[0].set_xlabel("x grid")
    axes[0].set_ylabel("y grid")

    axes[1].imshow(central_building, origin="upper", cmap="gray")
    axes[1].set_title("Central Building Mask")
    axes[1].set_xlabel("x grid")
    axes[1].set_ylabel("y grid")

    axes[2].imshow(mask, origin="upper", cmap="gray", alpha=0.5)
    axes[2].imshow(central_building, origin="upper", cmap="Reds", alpha=0.8)
    axes[2].set_title("Overlay: All Buildings + Central Building")
    axes[2].set_xlabel("x grid")
    axes[2].set_ylabel("y grid")

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # Plot 2: vector debug plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))

    bbox_projected.boundary.plot(
        ax=ax,
        color="red",
        linewidth=2,
        label="Search area",
    )

    if len(all_buildings) > 0:
        all_buildings.plot(
            ax=ax,
            color="lightgray",
            edgecolor="black",
            linewidth=0.5,
            label="All buildings",
        )

    if len(central_building_polygon) > 0:
        central_building_polygon.plot(
            ax=ax,
            color="black",
            edgecolor="yellow",
            linewidth=2,
            label="Central building",
        )

    center_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([lon], [lat]),
        crs="EPSG:4326",
    ).to_crs(result["crs_projected"])

    center_gdf.plot(
        ax=ax,
        color="red",
        markersize=40,
        label="Input lat/lon",
    )

    ax.set_title("Central Building Selection")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.legend()
    plt.tight_layout()
    plt.show()