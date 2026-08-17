"""
Filter the City of London Police 2024 street-level crime data down to
theft-family offences, for hotspot analysis.

Loads all monthly CSVs (reusing eda_london_crime_2024.load_data), drops
exact duplicate rows, filters to THEFT_CRIME_TYPES, drops rows with no
coordinates (can't be mapped), and saves the result as a GeoDataFrame
(EPSG:4326) to output/london_crime_2024/theft_crimes.geojson.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

from eda_london_crime_2024 import DATA_DIR, OUTPUT_DIR, load_data

# All 7 theft-family categories present in this dataset: the four direct
# theft offences, plus burglary/robbery/vehicle crime (theft plus another
# element -- unlawful entry, force, or a vehicle-specific offence).
THEFT_CRIME_TYPES = [
    "Other theft",
    "Theft from the person",
    "Shoplifting",
    "Bicycle theft",
    "Burglary",
    "Robbery",
    "Vehicle crime",
]


def filter_theft_crimes(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df):,} exact duplicate rows ({before:,} -> {len(df):,}).")

    theft_df = df[df["Crime type"].isin(THEFT_CRIME_TYPES)].copy()
    print(f"\nTheft-family rows: {len(theft_df):,} of {len(df):,} total ({len(theft_df) / len(df) * 100:.1f}%).")
    print("\nBy category:")
    print(theft_df["Crime type"].value_counts().to_string())

    no_geo = theft_df["Longitude"].isna().sum()
    theft_df = theft_df.dropna(subset=["Longitude", "Latitude"])
    print(
        f"\nDropped {no_geo:,} theft-family rows with no coordinates "
        f"('No Location') -- not usable for hotspot mapping."
    )
    print(f"Remaining, geocoded theft rows: {len(theft_df):,}")

    return theft_df


def to_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="EPSG:4326",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    theft_df = filter_theft_crimes(df)
    theft_gdf = to_geodataframe(theft_df)

    out_path = OUTPUT_DIR / "theft_crimes.geojson"
    theft_gdf.to_file(out_path, driver="GeoJSON")
    print(f"\nSaved {len(theft_gdf):,} theft-family incidents to {out_path}")


if __name__ == "__main__":
    main()
