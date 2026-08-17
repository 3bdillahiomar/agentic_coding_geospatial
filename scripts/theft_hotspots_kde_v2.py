"""
Map theft hotspots in the City of London using quartic kernel density
estimation, following the method in the NIJ report "Mapping Crime:
Understanding Hot Spots" (documents/hotspots.pdf, pp. 26-30):

- Cell size = shorter side of the point extent / 150 (Ratcliffe 1999b).
- Bandwidth = mean distance to the K-th nearest neighbor (K=8 by default --
  see the plan's Deviations section: the report's exact K-selection rule,
  cited to Williamson et al. 1999, isn't present in the extracted text).
- Quartic kernel, computed directly (no library ships this kernel).
- Grid cells outside the convex hull of the points are left unshaded.
- Legend: incremental-mean classification (0-mean, mean-2*mean, ...,
  >5*mean), the report's own recommended thematic threshold scheme.

Reads output/london_crime_2024/theft_crimes.geojson (via theft_hotspots_kde
for the loader/outlier-filter, reused unchanged) and saves
output/london_crime_2024/theft_hotspots_kde_v2.png. The original
Gaussian-kernel theft_hotspots_kde.py / .png are left untouched for
comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull, cKDTree

from theft_hotspots_kde import OUTPUT_DIR, INK_PRIMARY, INK_SECONDARY, SURFACE, load_points, drop_outliers

OUTPUT_PATH = OUTPUT_DIR / "theft_hotspots_kde_v2.png"

CELL_SIZE_DIVISOR = 150  # Ratcliffe 1999b
K_NEIGHBOR_ORDER = 8  # see module docstring -- an assumption, not from the text
BUFFER_M = 150

# Report's classic hot-cold ramp (Exhibits 14-17): blue/green (low) through
# yellow/orange to red/purple (highest class) -- a deliberate deviation from
# this project's usual sequential-blue chart palette, to follow the
# reference exactly for this map.
CLASS_COLORS = ["#e8f4ea", "#66c2a5", "#ffeb84", "#fdae61", "#e34a33", "#7a0177"]
CLASS_LABELS = [
    "0 to mean", "Mean to 2x mean", "2x to 3x mean",
    "3x to 4x mean", "4x to 5x mean", "Greater than 5x mean",
]


def cell_size_from_extent(x: np.ndarray, y: np.ndarray) -> float:
    shorter_side = min(x.max() - x.min(), y.max() - y.min())
    return shorter_side / CELL_SIZE_DIVISOR


def bandwidth_from_knn(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """Mean distance to the K-th nearest neighbor, used as the KDE bandwidth.

    Computed on *unique* coordinates, not the raw points. police.uk snaps
    crime locations to a small set of anonymized points per street/LSOA --
    98.9% of these theft points share an exact coordinate with another
    point (5,514 points collapse to 284 unique locations). Using the raw
    points would make the K-nearest-neighbor distance near-zero for almost
    every point (a duplicate at distance 0 dominates), collapsing the
    bandwidth below the cell size and breaking the KDE surface. This wasn't
    a factor in the report's original data and is flagged as an additional
    deviation driven by this dataset's anonymization, not a reference
    ambiguity.
    """
    unique_points = np.unique(np.column_stack([x, y]), axis=0)
    tree = cKDTree(unique_points)
    distances, _ = tree.query(unique_points, k=k + 1)  # +1: point itself is its own 0th neighbor
    return distances[:, k].mean()


def build_grid(x: np.ndarray, y: np.ndarray, cell_size: float):
    xmin, xmax = x.min() - BUFFER_M, x.max() + BUFFER_M
    ymin, ymax = y.min() - BUFFER_M, y.max() + BUFFER_M
    xs = np.arange(xmin, xmax, cell_size)
    ys = np.arange(ymin, ymax, cell_size)
    xx, yy = np.meshgrid(xs, ys)
    return xx, yy


def quartic_kde(points: np.ndarray, xx: np.ndarray, yy: np.ndarray, bandwidth: float) -> np.ndarray:
    """Quartic (biweight) kernel density, summed per grid cell.

    weight(d) = (3 / (pi * r^2)) * (1 - (d/r)^2)^2  for d <= r, else 0.
    """
    point_tree = cKDTree(points)
    grid_coords = np.column_stack([xx.ravel(), yy.ravel()])
    density = np.zeros(grid_coords.shape[0])

    neighbor_lists = point_tree.query_ball_point(grid_coords, r=bandwidth)
    coeff = 3 / (np.pi * bandwidth**2)
    for i, neighbors in enumerate(neighbor_lists):
        if not neighbors:
            continue
        d = np.linalg.norm(points[neighbors] - grid_coords[i], axis=1)
        density[i] = coeff * np.sum((1 - (d / bandwidth) ** 2) ** 2)

    return density.reshape(xx.shape)


def mask_outside_hull(xx: np.ndarray, yy: np.ndarray, points: np.ndarray) -> np.ndarray:
    hull = ConvexHull(points)
    hull_path = MplPath(points[hull.vertices])
    grid_coords = np.column_stack([xx.ravel(), yy.ravel()])
    inside = hull_path.contains_points(grid_coords)
    return inside.reshape(xx.shape)


def classify_incremental_mean(density: np.ndarray, in_hull: np.ndarray) -> np.ndarray:
    """6-class incremental-mean legend, per the report (pp. 29-30).

    The mean is computed only over in-hull cells with density > 0.
    """
    eligible = in_hull & (density > 0)
    mean_value = density[eligible].mean()
    print(f"Grid-cell mean density (in-hull, nonzero cells): {mean_value:.2e} (quartic kernel units, m^-2)")

    edges = [0, mean_value, 2 * mean_value, 3 * mean_value, 4 * mean_value, 5 * mean_value, np.inf]
    classified = np.digitize(density, edges[1:], right=False)  # 0..5
    classified = np.where(in_hull, classified, -1)  # -1 marks outside-hull (unshaded)
    return classified


def plot_hotspots(gdf, xx, yy, classified: np.ndarray, cell_size: float, bandwidth: float) -> None:
    cmap = ListedColormap(CLASS_COLORS)
    cmap.set_bad(alpha=0)
    masked = np.ma.masked_where(classified < 0, classified)
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(9, 8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    mesh = ax.pcolormesh(xx, yy, masked, cmap=cmap, norm=norm, shading="nearest")
    ax.scatter(gdf.geometry.x, gdf.geometry.y, s=4, color=INK_PRIMARY, alpha=0.2, linewidths=0)

    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m, EPSG:27700)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Northing (m, EPSG:27700)", color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.04, pad=0.03, ticks=range(6))
    cbar.ax.set_yticklabels(CLASS_LABELS, fontsize=8, color=INK_SECONDARY)
    cbar.outline.set_visible(False)

    ax.set_title(
        f"Theft hotspots (quartic KDE) - City of London, 2024\n"
        f"{len(gdf):,} incidents, cell size {cell_size:.0f} m, "
        f"bandwidth {bandwidth:.0f} m (K={K_NEIGHBOR_ORDER})",
        fontsize=12, color=INK_PRIMARY, loc="left", pad=12,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)


def main() -> None:
    gdf = load_points()
    gdf = drop_outliers(gdf)
    x, y = gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()
    points = np.column_stack([x, y])

    cell_size = cell_size_from_extent(x, y)
    bandwidth = bandwidth_from_knn(x, y, K_NEIGHBOR_ORDER)
    print(f"Cell size (shorter extent / {CELL_SIZE_DIVISOR}): {cell_size:.1f} m")
    print(f"Bandwidth (mean distance to {K_NEIGHBOR_ORDER}th nearest neighbor): {bandwidth:.1f} m")

    xx, yy = build_grid(x, y, cell_size)
    print(f"Grid: {xx.shape[0]} x {xx.shape[1]} cells")

    density = quartic_kde(points, xx, yy, bandwidth)
    in_hull = mask_outside_hull(xx, yy, points)
    classified = classify_incremental_mean(density, in_hull)

    plot_hotspots(gdf, xx, yy, classified, cell_size, bandwidth)
    print(f"\nSaved hotspot map to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
