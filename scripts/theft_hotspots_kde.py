"""
Map theft hotspots in the City of London with a Kernel Density Estimate
(KDE) surface.

Loads output/london_crime_2024/theft_crimes.geojson (produced by
filter_theft_crimes.py), reprojects to British National Grid (EPSG:27700)
so the KDE bandwidth is isotropic in metres, fits a Gaussian KDE over the
theft locations, and renders the density surface with the incident points
overlaid as output/london_crime_2024/theft_hotspots_kde.png.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

from eda_london_crime_2024 import OUTPUT_DIR

INPUT_PATH = OUTPUT_DIR / "theft_crimes.geojson"
OUTPUT_PATH = OUTPUT_DIR / "theft_hotspots_kde.png"

# British National Grid -- metres, appropriate for a small, local KDE bandwidth.
PROJECTED_CRS = "EPSG:27700"
GRID_RESOLUTION = 250  # cells per axis
BUFFER_M = 150  # padding around the point extent, in metres
N_TOP_HOTSPOTS = 5
# Points farther than this from the median location are treated as
# geocoding outliers (see drop_outliers) and excluded from the KDE.
OUTLIER_DISTANCE_M = 3000

# Dataviz palette: sequential blue ramp, light -> dark.
SEQUENTIAL_RAMP = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"


def load_points() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(INPUT_PATH)
    return gdf.to_crs(PROJECTED_CRS)


def drop_outliers(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop points far from the core cluster.

    A handful of rows carry coordinates several km away in other boroughs
    (e.g. Enfield, Newham) despite being recorded by City of London Police --
    likely geocoding/jurisdiction artifacts. Left in, they blow up the KDE
    grid extent and bury the real hotspot detail in one corner of the map.
    """
    x, y = gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()
    median_point = np.median(x), np.median(y)
    distance = np.hypot(x - median_point[0], y - median_point[1])
    is_outlier = distance > OUTLIER_DISTANCE_M

    if is_outlier.any():
        print(
            f"\nExcluding {is_outlier.sum()} point(s) more than "
            f"{OUTLIER_DISTANCE_M:,} m from the median theft location "
            "(likely geocoding/jurisdiction outliers):"
        )
        for _, row in gdf.loc[is_outlier].iterrows():
            print(f"  - {row['Crime type']}: {row['Location']} ({row['LSOA name']})")

    return gdf.loc[~is_outlier].copy()


def fit_kde(x: np.ndarray, y: np.ndarray) -> gaussian_kde:
    kde = gaussian_kde(np.vstack([x, y]))
    bandwidth_m = kde.factor * np.mean([x.std(), y.std()])
    print(f"KDE bandwidth (Scott's rule): ~{bandwidth_m:.0f} m")
    return kde


def evaluate_grid(kde: gaussian_kde, x: np.ndarray, y: np.ndarray, n_points: int):
    xmin, xmax = x.min() - BUFFER_M, x.max() + BUFFER_M
    ymin, ymax = y.min() - BUFFER_M, y.max() + BUFFER_M
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, GRID_RESOLUTION),
        np.linspace(ymin, ymax, GRID_RESOLUTION),
    )
    positions = np.vstack([xx.ravel(), yy.ravel()])
    density = kde(positions).reshape(xx.shape)
    # KDE integrates to 1 over the plane in units of 1/m^2; scale by the
    # point count and convert to incidents per km^2 for a readable colorbar.
    density_per_km2 = density * n_points * 1e6
    return xx, yy, density_per_km2


def print_top_hotspots(gdf: gpd.GeoDataFrame, xx, yy, density_per_km2) -> None:
    flat_idx = np.argsort(density_per_km2.ravel())[::-1]
    print(f"\nTop {N_TOP_HOTSPOTS} density peaks (nearest recorded location shown):")
    seen_cells = []
    shown = 0
    for idx in flat_idx:
        if shown >= N_TOP_HOTSPOTS:
            break
        row, col = np.unravel_index(idx, density_per_km2.shape)
        px, py = xx[row, col], yy[row, col]
        # Skip near-duplicate peaks that are within one grid cell of a peak already reported.
        cell_w = (xx[0, 1] - xx[0, 0])
        if any(abs(px - sx) < cell_w * 3 and abs(py - sy) < cell_w * 3 for sx, sy in seen_cells):
            continue
        seen_cells.append((px, py))
        distances = np.hypot(gdf.geometry.x - px, gdf.geometry.y - py)
        nearest = gdf.iloc[distances.idxmin()]
        print(
            f"  {shown + 1}. {density_per_km2[row, col]:,.0f} incidents/km^2 "
            f"near '{nearest['Location']}'"
        )
        shown += 1


def plot_hotspots(gdf: gpd.GeoDataFrame, xx, yy, density_per_km2) -> None:
    cmap = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL_RAMP)

    fig, ax = plt.subplots(figsize=(9, 8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    contour = ax.contourf(xx, yy, density_per_km2, levels=15, cmap=cmap)
    ax.scatter(
        gdf.geometry.x, gdf.geometry.y,
        s=5, color=INK_PRIMARY, alpha=0.25, linewidths=0,
    )

    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m, EPSG:27700)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Northing (m, EPSG:27700)", color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(contour, ax=ax, fraction=0.04, pad=0.03)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    cbar.set_label("Estimated theft density (incidents / km^2)", color=INK_SECONDARY, fontsize=9)

    ax.set_title(
        f"Theft hotspots (KDE) - City of London, 2024\n"
        f"{len(gdf):,} theft-family incidents across 7 categories",
        fontsize=12, color=INK_PRIMARY, loc="left", pad=12,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)


def main() -> None:
    gdf = load_points()
    gdf = drop_outliers(gdf)
    x, y = gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()

    print(f"Fitting KDE on {len(gdf):,} theft incidents.")
    kde = fit_kde(x, y)
    xx, yy, density_per_km2 = evaluate_grid(kde, x, y, n_points=len(gdf))

    print_top_hotspots(gdf, xx, yy, density_per_km2)
    plot_hotspots(gdf, xx, yy, density_per_km2)
    print(f"\nSaved hotspot map to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
