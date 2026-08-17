# Conversation Summary: London Crime 2024 Theft Hotspot Analysis

Date: 2026-08-17

## Sequence of requests and outcomes

1. **Exploratory data analysis** of `data/london_crime_2024/` — row/column summary, missing-data flags, data quality issues, charts of crime types and monthly patterns.
   - Script: `scripts/eda_london_crime_2024.py`
   - Outputs: `output/london_crime_2024/crime_types.png`, `monthly_trend.png`, `monthly_by_type_heatmap.png`
   - Quality flags: empty `Context` column; ASB rows structurally lack `Crime ID`/outcome; ~9.6% of rows lack coordinates; 154 exact duplicate rows; `Crime ID` not a reliable unique key; single-force dataset.

2. **Theft hotspot category selection and filtering.** User chose all 7 theft-family categories (Other theft, Theft from the person, Shoplifting, Bicycle theft, Burglary, Robbery, Vehicle crime).
   - Script: `scripts/filter_theft_crimes.py`
   - Output: `output/london_crime_2024/theft_crimes.geojson` (5,522 rows)

3. **Technique selection for hotspot mapping** — recommended and selected Kernel Density Estimation (KDE).

4. **Gaussian KDE static map** (matplotlib + `scipy.stats.gaussian_kde`).
   - Script: `scripts/theft_hotspots_kde.py`
   - Output: `output/london_crime_2024/theft_hotspots_kde.png`
   - Discovered and fixed a geographic outlier issue (8 points several km from the core cluster) via `drop_outliers()`, excluding points >3,000 m from the median location.

5. **Alignment with reference methodology** — `documents/hotspots.pdf` (NIJ Special Report, "Mapping Crime: Understanding Hot Spots," Spencer Chainey). Plan built and approved (see `~/.claude/plans/i-would-like-to-vast-lerdorf.md`), specifying:
   - Preliminary global statistics (mean center, standard deviation ellipse, Nearest Neighbor Index, Moran's I / Geary's C) before trusting a hotspot map.
   - Quartic (biweight) KDE with Ratcliffe 1999b cell-size rule and K-nearest-neighbor bandwidth, saved as a new `_v2` script/output so the original Gaussian KDE stays untouched for comparison.
   - An explicit, itemized list of deviations from the reference (no library quartic kernel, assumed K=8 bandwidth order, convex hull as study-area proxy, per-map color scheme change, no Gi* LISA, no rate/denominator map, combined not per-category stats, no space-time animation).

   - Script: `scripts/theft_hotspot_stats.py`
     - Output: `output/london_crime_2024/theft_mean_center_ellipse.png`
     - Results: mean center (532647.9, 181243.3) EPSG:27700; std dev distance 757.4 m; NNI = 0.079 (z = -130.9, n = 5,514); Moran's I = 0.1055 (z = 5.2); Geary's C = 0.9269 (z = -1.9) — both tests support clustering.
   - Script: `scripts/theft_hotspots_kde_v2.py`
     - Output: `output/london_crime_2024/theft_hotspots_kde_v2.png`
     - Cell size 20.2 m, bandwidth 253.1 m (K=8), grid 165x267.
     - Fixed a bandwidth-collapse bug caused by police.uk coordinate anonymization (98.9% of points share exact coordinates with another point) by computing K-NN bandwidth on de-duplicated coordinates only.

6. **Hotspot polygon extraction.** Criterion: "Greater than 5x mean" grid cells, merged and converted to polygons.
   - Script: `scripts/extract_theft_hotspot_polygons.py`
   - Output: `output/london_crime_2024/london_theft_hotspots.geojson` (4 polygons, EPSG:4326)

   | hotspot_id | cell_count | area_m2 | incident_count |
   |---|---|---|---|
   | 2 | 267 | 108,443 | 794 |
   | 4 | 222 | 90,166 | 620 |
   | 1 | 215 | 87,323 | 469 |
   | 3 | 12 | 4,874 | 93 |

## Key files produced

```
scripts/
  eda_london_crime_2024.py
  filter_theft_crimes.py
  theft_hotspots_kde.py          # v1: Gaussian KDE (kept unmodified for comparison)
  theft_hotspot_stats.py         # NIJ preliminary global statistics
  theft_hotspots_kde_v2.py       # v2: Quartic KDE per NIJ reference method
  extract_theft_hotspot_polygons.py

output/london_crime_2024/
  crime_types.png
  monthly_trend.png
  monthly_by_type_heatmap.png
  theft_crimes.geojson
  theft_hotspots_kde.png
  theft_mean_center_ellipse.png
  theft_hotspots_kde_v2.png
  london_theft_hotspots.geojson
```

## Notable technical decisions

- Conda environment: `claude_code_geoai` (pandas, geopandas, scipy, shapely, rasterio, matplotlib, mapclassify). `esda`/`libpysal`/`h3`/`seaborn`/`contextily` are not installed.
- EPSG:27700 (British National Grid) used for all distance-based analysis; EPSG:4326 used for GeoJSON I/O.
- Manual implementations (no new dependencies) for: quartic kernel density, rook-contiguity weights matrix, Moran's I / Geary's C with permutation testing.
- police.uk location anonymization (coordinate snapping to a small set of fixed points) required de-duplicating coordinates before computing K-nearest-neighbor bandwidth — otherwise zero-distance duplicate pairs collapsed the bandwidth below the grid cell size.

## Open item (not yet actioned)

The Nearest Neighbor Index in `theft_hotspot_stats.py` still runs on raw (non-deduplicated) points. Its very low NNI / strongly negative z-score is directionally correct evidence of clustering, but is partly inflated by the same coordinate-snapping artifact that affected the v2 KDE bandwidth. No fix has been requested or applied.
