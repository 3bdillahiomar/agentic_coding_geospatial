"""
Preliminary global statistics for theft hotspot analysis, following the
"Preliminary global statistical tests" section of the NIJ report
"Mapping Crime: Understanding Hot Spots" (documents/hotspots.pdf, pp. 15-19).

Computes, for the combined theft-family points (output/london_crime_2024/
theft_crimes.geojson, same outlier-filtered set used by the KDE scripts):

- Mean center and standard deviation distance/ellipse (saved as
  output/london_crime_2024/theft_mean_center_ellipse.png)
- Nearest Neighbor Index (NNI) and z-score, testing for clustering
- Moran's I and Geary's C on a coarse quadrat aggregation, testing for
  spatial autocorrelation

The report's own conclusion (p. 33) is that these tests should come before
building a hot spot map -- if they show no clustering, a hot spot map isn't
worth building. This script exists to make that check explicit.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull, cKDTree

from theft_hotspots_kde import OUTPUT_DIR, INK_PRIMARY, INK_SECONDARY, SURFACE, load_points, drop_outliers

ELLIPSE_PATH = OUTPUT_DIR / "theft_mean_center_ellipse.png"
# Quadrat cell size for Moran's I / Geary's C: coarser than the KDE surface
# grid so cell counts aren't dominated by zeros.
QUADRAT_DIVISOR = 30
N_PERMUTATIONS = 999
RANDOM_SEED = 0


def mean_center_and_dispersion(x: np.ndarray, y: np.ndarray):
    mean_x, mean_y = x.mean(), y.mean()
    dx, dy = x - mean_x, y - mean_y

    std_distance = np.sqrt(np.mean(dx**2 + dy**2))

    cov = np.cov(np.vstack([dx, dy]), ddof=0)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    semi_major, semi_minor = np.sqrt(eigenvalues[0]), np.sqrt(eigenvalues[1])
    angle_deg = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    return {
        "mean_x": mean_x, "mean_y": mean_y, "std_distance": std_distance,
        "semi_major": semi_major, "semi_minor": semi_minor, "angle_deg": angle_deg,
    }


def plot_mean_center_ellipse(x: np.ndarray, y: np.ndarray, stats: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.scatter(x, y, s=4, color=INK_SECONDARY, alpha=0.25, linewidths=0)
    ax.scatter([stats["mean_x"]], [stats["mean_y"]], s=80, color=INK_PRIMARY,
               marker="+", linewidths=2, zorder=3, label="Mean center")

    ellipse = Ellipse(
        (stats["mean_x"], stats["mean_y"]),
        width=2 * stats["semi_major"], height=2 * stats["semi_minor"],
        angle=stats["angle_deg"], facecolor="none", edgecolor="#e34948",
        linewidth=2, zorder=2, label="1 std-dev ellipse",
    )
    ax.add_patch(ellipse)

    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m, EPSG:27700)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Northing (m, EPSG:27700)", color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK_SECONDARY)

    ax.set_title(
        "Theft mean center and dispersion - City of London, 2024",
        fontsize=12, color=INK_PRIMARY, loc="left", pad=12,
    )
    fig.tight_layout()
    fig.savefig(ELLIPSE_PATH, dpi=150)
    plt.close(fig)


def nearest_neighbor_index(x: np.ndarray, y: np.ndarray) -> dict:
    n = len(x)
    points = np.column_stack([x, y])
    tree = cKDTree(points)
    # k=2: nearest neighbor after the point itself (distance 0 to itself).
    distances, _ = tree.query(points, k=2)
    observed_mean_nnd = distances[:, 1].mean()

    area = ConvexHull(points).volume  # scipy quirk: 'volume' is area in 2D
    expected_mean_nnd = 0.5 * np.sqrt(area / n)
    se = 0.26136 / np.sqrt(n**2 / area)
    nni = observed_mean_nnd / expected_mean_nnd
    z_score = (observed_mean_nnd - expected_mean_nnd) / se

    return {
        "n": n, "area_km2": area / 1e6, "observed_mean_nnd": observed_mean_nnd,
        "expected_mean_nnd": expected_mean_nnd, "nni": nni, "z_score": z_score,
    }


def quadrat_counts(x: np.ndarray, y: np.ndarray):
    hull = ConvexHull(np.column_stack([x, y]))
    hull_path = hull.points[hull.vertices]

    cell_size = min(x.max() - x.min(), y.max() - y.min()) / QUADRAT_DIVISOR
    n_cols = int(np.ceil((x.max() - x.min()) / cell_size)) + 1
    n_rows = int(np.ceil((y.max() - y.min()) / cell_size)) + 1

    col_idx = ((x - x.min()) / cell_size).astype(int)
    row_idx = ((y - y.min()) / cell_size).astype(int)
    counts = np.zeros((n_rows, n_cols), dtype=int)
    np.add.at(counts, (row_idx, col_idx), 1)

    from matplotlib.path import Path as MplPath
    cell_centers_x = x.min() + (np.arange(n_cols) + 0.5) * cell_size
    cell_centers_y = y.min() + (np.arange(n_rows) + 0.5) * cell_size
    cx, cy = np.meshgrid(cell_centers_x, cell_centers_y)
    in_hull = MplPath(hull_path).contains_points(np.column_stack([cx.ravel(), cy.ravel()]))
    in_hull = in_hull.reshape(counts.shape)

    return counts, in_hull, cell_size


def rook_weights(shape, mask: np.ndarray) -> np.ndarray:
    """Binary rook-contiguity weights matrix over the masked (in-hull) cells."""
    rows, cols = np.where(mask)
    cell_ids = {(r, c): i for i, (r, c) in enumerate(zip(rows, cols))}
    n_cells = len(cell_ids)
    w = np.zeros((n_cells, n_cells))
    for (r, c), i in cell_ids.items():
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (r + dr, c + dc)
            if neighbor in cell_ids:
                w[i, cell_ids[neighbor]] = 1
    return w


def morans_i_and_gearys_c(values: np.ndarray, w: np.ndarray, rng: np.random.Generator) -> dict:
    n = len(values)
    s0 = w.sum()
    mean_v = values.mean()
    dev = values - mean_v

    num_i = (w * np.outer(dev, dev)).sum()
    denom = (dev**2).sum()
    moran_i = (n / s0) * (num_i / denom)

    diff_sq = (values[:, None] - values[None, :]) ** 2
    num_c = (w * diff_sq).sum()
    geary_c = ((n - 1) / (2 * s0)) * (num_c / denom)

    perm_i, perm_c = np.empty(N_PERMUTATIONS), np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        shuffled = rng.permutation(values)
        dev_s = shuffled - shuffled.mean()
        perm_i[i] = (n / s0) * ((w * np.outer(dev_s, dev_s)).sum() / (dev_s**2).sum())
        diff_sq_s = (shuffled[:, None] - shuffled[None, :]) ** 2
        perm_c[i] = ((n - 1) / (2 * s0)) * ((w * diff_sq_s).sum() / (dev_s**2).sum())

    z_i = (moran_i - perm_i.mean()) / perm_i.std()
    z_c = (geary_c - perm_c.mean()) / perm_c.std()

    return {"moran_i": moran_i, "moran_z": z_i, "geary_c": geary_c, "geary_z": z_c}


def main() -> None:
    gdf = load_points()
    gdf = drop_outliers(gdf)
    x, y = gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()

    print("=" * 70)
    print("MEAN CENTER AND DISPERSION")
    print("=" * 70)
    disp = mean_center_and_dispersion(x, y)
    print(f"Mean center: ({disp['mean_x']:.1f}, {disp['mean_y']:.1f}) [EPSG:27700]")
    print(f"Standard deviation distance: {disp['std_distance']:.1f} m")
    print(
        f"Standard deviation ellipse: semi-major {disp['semi_major']:.1f} m, "
        f"semi-minor {disp['semi_minor']:.1f} m, orientation {disp['angle_deg']:.1f} deg"
    )
    plot_mean_center_ellipse(x, y, disp)
    print(f"Saved {ELLIPSE_PATH}")

    print("\n" + "=" * 70)
    print("NEAREST NEIGHBOR INDEX (Clark-Evans)")
    print("=" * 70)
    nni = nearest_neighbor_index(x, y)
    print(f"n = {nni['n']:,}, convex hull area = {nni['area_km2']:.3f} km^2")
    print(f"Observed mean nearest-neighbor distance: {nni['observed_mean_nnd']:.1f} m")
    print(f"Expected (CSR) mean nearest-neighbor distance: {nni['expected_mean_nnd']:.1f} m")
    print(f"NNI = {nni['nni']:.3f}  (z = {nni['z_score']:.1f})")
    if nni["nni"] < 1 and nni["z_score"] < -1.96:
        print("-> NNI < 1 with a significantly negative z-score: theft points are clustered.")
    else:
        print("-> No significant evidence of clustering from the NNI test.")

    print("\n" + "=" * 70)
    print("MORAN'S I / GEARY'S C (quadrat aggregation, rook contiguity)")
    print("=" * 70)
    counts, in_hull, cell_size = quadrat_counts(x, y)
    values = counts[in_hull].astype(float)
    w = rook_weights(counts.shape, in_hull)
    print(f"Quadrat cell size: {cell_size:.1f} m, {in_hull.sum():,} in-hull cells")
    rng = np.random.default_rng(RANDOM_SEED)
    auto = morans_i_and_gearys_c(values, w, rng)
    print(f"Moran's I = {auto['moran_i']:.4f}  (permutation z = {auto['moran_z']:.1f})")
    print(f"Geary's C = {auto['geary_c']:.4f}  (permutation z = {auto['geary_z']:.1f})")
    if auto["moran_i"] > 0 and auto["moran_z"] > 1.96:
        print("-> Positive, significant Moran's I: theft counts are spatially autocorrelated (clustered).")
    else:
        print("-> No significant positive spatial autocorrelation detected.")

    clustered = nni["nni"] < 1 and nni["z_score"] < -1.96 and auto["moran_i"] > 0 and auto["moran_z"] > 1.96
    print("\n" + "=" * 70)
    if clustered:
        print("Both tests indicate clustering -> proceeding to hot spot mapping is justified.")
    else:
        print("Not all tests indicate significant clustering -> review results before hot spot mapping.")
    print("=" * 70)


if __name__ == "__main__":
    main()
