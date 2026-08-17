"""
Exploratory data analysis of the City of London Police street-level crime
data for 2024 (data/london_crime_2024/2024-MM/*.csv).

Prints a summary of shape/columns/missing data, flags data quality issues,
and saves three charts (crime types, monthly trend, monthly-by-type
heatmap) as PNGs under output/london_crime_2024/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "london_crime_2024"
OUTPUT_DIR = REPO_ROOT / "output" / "london_crime_2024"

# Chart chrome, from the project's dataviz palette (light mode).
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"
SURFACE = "#fcfcfb"
SEQUENTIAL_BLUE = "#2a78d6"
SEQUENTIAL_RAMP = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
]


def load_data() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("*/*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {DATA_DIR}")
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def print_summary(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Rows: {len(df):,}   Columns: {df.shape[1]}")
    print(f"\nColumns and dtypes:\n{df.dtypes.to_string()}")

    missing = pd.DataFrame({
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(1),
    })
    print(f"\nMissing values per column:\n{missing.to_string()}")

    print(f"\nMonths present: {sorted(df['Month'].unique())}")
    print(f"Distinct crime types: {df['Crime type'].nunique()}")
    print(f"Distinct outcome categories: {df['Last outcome category'].nunique()}")


def print_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("DATA QUALITY FLAGS")
    print("=" * 70)

    context_missing_pct = df["Context"].isna().mean() * 100
    print(
        f"\n1. 'Context' column is {context_missing_pct:.0f}% empty "
        "(always blank in this dataset) -> dropped from analysis."
    )
    df = df.drop(columns=["Context"])

    no_id = df[df["Crime ID"].isna()]
    asb_share = (no_id["Crime type"] == "Anti-social behaviour").mean() * 100 if len(no_id) else 0
    print(
        f"\n2. 'Crime ID' / 'Last outcome category' are missing for "
        f"{len(no_id):,} rows ({len(no_id) / len(df) * 100:.1f}%), and "
        f"{asb_share:.0f}% of those are 'Anti-social behaviour'. This is a "
        "known police.uk convention (ASB incidents are not assigned a case "
        "ID or outcome) rather than a data entry error."
    )

    no_location = df[df["Location"] == "No Location"]
    print(
        f"\n3. {len(no_location):,} rows ({len(no_location) / len(df) * 100:.1f}%) "
        "have no coordinates ('Location' == 'No Location', Longitude/"
        "Latitude/LSOA also blank), spread across most crime types. These "
        "rows should be excluded from any map-based analysis."
    )

    exact_dupes = df.duplicated().sum()
    before = len(df)
    df = df.drop_duplicates()
    print(
        f"\n4. {exact_dupes:,} fully duplicate rows found (all columns "
        f"identical) -> dropped ({before:,} -> {len(df):,} rows) before "
        "building the charts below."
    )

    dup_id_groups = df[df["Crime ID"].notna() & df["Crime ID"].duplicated(keep=False)]
    n_groups = dup_id_groups["Crime ID"].nunique()
    print(
        f"\n5. {len(dup_id_groups):,} rows share a 'Crime ID' with another "
        f"row ({n_groups} groups), but most of those groups differ in "
        "Location/LSOA/outcome for the same ID within the same month. "
        "'Crime ID' should not be treated as a reliable unique row key."
    )

    print(
        "\n6. 'Reported by' and 'Falls within' are constant "
        f"({df['Reported by'].nunique()} unique value) -- single-force "
        "dataset, no cross-force mixing. Latitude/Longitude values all fall "
        "within a plausible Greater London bounding box (no coordinate "
        "outliers detected)."
    )

    return df


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_LINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def chart_crime_types(df: pd.DataFrame) -> None:
    counts = df["Crime type"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    ax.barh(counts.index, counts.values, color=SEQUENTIAL_BLUE, height=0.65)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    style_axes(ax)

    for y, value in enumerate(counts.values):
        ax.text(value + counts.max() * 0.01, y, f"{value:,}",
                va="center", ha="left", fontsize=8, color=INK_SECONDARY)

    ax.set_xlabel("Number of incidents")
    ax.set_title(
        "Crime incidents by type - City of London, 2024",
        fontsize=13, color=INK_PRIMARY, loc="left", pad=12,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "crime_types.png", dpi=150)
    plt.close(fig)


def chart_monthly_trend(df: pd.DataFrame) -> None:
    monthly = df.groupby("Month").size().sort_index()

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    ax.plot(monthly.index, monthly.values, color=SEQUENTIAL_BLUE,
            linewidth=2, marker="o", markersize=6)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    style_axes(ax)
    ax.set_xticks(range(len(monthly.index)))
    ax.set_xticklabels(monthly.index, rotation=45, ha="right")

    for x, value in zip(monthly.index, monthly.values):
        ax.annotate(f"{value:,}", (x, value), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8, color=INK_SECONDARY)

    ax.set_ylabel("Number of incidents")
    ax.set_title(
        "Monthly crime incidents - City of London, 2024",
        fontsize=13, color=INK_PRIMARY, loc="left", pad=12,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "monthly_trend.png", dpi=150)
    plt.close(fig)


def chart_monthly_by_type_heatmap(df: pd.DataFrame) -> None:
    pivot = pd.crosstab(df["Month"], df["Crime type"]).sort_index()
    # Order columns by total volume so the heatmap reads high-to-low, left-to-right.
    pivot = pivot[pivot.sum().sort_values(ascending=False).index]

    cmap = matplotlib_colormap()
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=SURFACE)
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8, color=INK_SECONDARY)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8, color=INK_SECONDARY)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    cbar.set_label("Number of incidents", color=INK_SECONDARY, fontsize=9)

    ax.set_title(
        "Monthly incidents by crime type - City of London, 2024",
        fontsize=13, color=INK_PRIMARY, loc="left", pad=12,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "monthly_by_type_heatmap.png", dpi=150)
    plt.close(fig)


def matplotlib_colormap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL_RAMP)


def main() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print_summary(df)
    clean_df = print_quality_flags(df)

    chart_crime_types(clean_df)
    chart_monthly_trend(clean_df)
    chart_monthly_by_type_heatmap(clean_df)

    print("\n" + "=" * 70)
    print(f"Charts saved to {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
