"""
Extract the "Greater than 5x mean" grid cells from the quartic KDE theft
hotspot surface (theft_hotspots_kde_v2.py) and convert them to polygons.

Reuses the exact same cell size / bandwidth / quartic KDE / incremental-mean
classification pipeline as theft_hotspots_kde_v2.py, so the polygons match
the top legend class shown on output/london_crime_2024/theft_hotspots_kde_v2.png.
Adjacent qualifying cells are merged into single polygons (rasterio,
4-connectivity), each attributed with its area, grid-cell count, peak/mean
kernel density, and the count of theft incidents it contains.

Saves output/london_crime_2024/london_theft_hotspots.geojson in EPSG:4326.
"""

import geopandas as gpd
import numpy as np
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import shape

from theft_hotspots_kde import OUTPUT_DIR, load_points, drop_outliers
from theft_hotspots_kde_v2 import (
    K_NEIGHBOR_ORDER,
    cell_size_from_extent,
    bandwidth_from_knn,
    build_grid,
    quartic_kde,
    mask_outside_hull,
    classify_incremental_mean,
)

OUTPUT_PATH = OUTPUT_DIR / "london_theft_hotspots.geojson"
PROJECTED_CRS = "EPSG:27700"
TOP_CLASS = 5  # "Greater than 5x mean" -- see CLASS_LABELS in theft_hotspots_kde_v2.py


def polygonize_top_class(classified: np.ndarray, xx: np.ndarray, yy: np.ndarray, cell_size: float) -> gpd.GeoDataFrame:
    is_hotspot = (classified == TOP_CLASS).astype(np.uint8)

    # xx/yy hold cell *centers*; shift to the lower-left corner of cell (0, 0)
    # to build the affine transform rasterio.features.shapes expects.
    corner_x0 = xx[0, 0] - cell_size / 2
    corner_y0 = yy[0, 0] - cell_size / 2
    transform = Affine(cell_size, 0, corner_x0, 0, cell_size, corner_y0)

    polygons = [
        shape(geom) for geom, value in shapes(is_hotspot, mask=is_hotspot.astype(bool), transform=transform)
        if value == 1
    ]
    return gpd.GeoDataFrame({"hotspot_id": range(1, len(polygons) + 1)}, geometry=polygons, crs=PROJECTED_CRS)


def attribute_hotspots(hotspots: gpd.GeoDataFrame, xx, yy, density, classified, points_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    is_top = classified == TOP_CLASS
    cell_points = gpd.GeoDataFrame(
        {"density": density[is_top]},
        geometry=gpd.points_from_xy(xx[is_top], yy[is_top]),
        crs=PROJECTED_CRS,
    )
    cells_joined = gpd.sjoin(cell_points, hotspots, predicate="within", how="inner")
    cell_stats = cells_joined.groupby("hotspot_id")["density"].agg(
        cell_count="count", mean_density="mean", max_density="max",
    )

    incidents_joined = gpd.sjoin(points_gdf, hotspots, predicate="within", how="inner")
    incident_counts = incidents_joined.groupby("hotspot_id").size().rename("incident_count")

    hotspots = hotspots.merge(cell_stats, on="hotspot_id", how="left")
    hotspots = hotspots.merge(incident_counts, on="hotspot_id", how="left")
    hotspots["incident_count"] = hotspots["incident_count"].fillna(0).astype(int)
    hotspots["area_m2"] = hotspots.geometry.area
    return hotspots.sort_values("incident_count", ascending=False).reset_index(drop=True)


def main() -> None:
    gdf = load_points()
    gdf = drop_outliers(gdf)
    x, y = gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()
    points = np.column_stack([x, y])

    cell_size = cell_size_from_extent(x, y)
    bandwidth = bandwidth_from_knn(x, y, K_NEIGHBOR_ORDER)
    xx, yy = build_grid(x, y, cell_size)
    density = quartic_kde(points, xx, yy, bandwidth)
    in_hull = mask_outside_hull(xx, yy, points)
    classified = classify_incremental_mean(density, in_hull)

    hotspots = polygonize_top_class(classified, xx, yy, cell_size)
    print(f"Found {len(hotspots)} '{TOP_CLASS}x mean' hotspot polygon(s) before attribution.")

    hotspots = attribute_hotspots(hotspots, xx, yy, density, classified, gdf)
    print(hotspots[["hotspot_id", "cell_count", "area_m2", "incident_count"]].to_string(index=False))

    hotspots = hotspots.to_crs("EPSG:4326")
    hotspots.to_file(OUTPUT_PATH, driver="GeoJSON")
    print(f"\nSaved {len(hotspots)} hotspot polygons to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
