# Agentic Coding for Geospatial

A Python-based geospatial analysis project for exploring spatial data workflows, automation, visualization, and GeoAI-assisted development.

The repository currently includes examples for London crime analysis, theft hotspot detection, spatial statistics, and geospatial visualization.

## Project Structure

```text
agentic_coding_geospatial/
├── data/
│   ├── chirps/
│   ├── london_crime_2024/
│   └── route_optimization/
├── documents/
├── output/
├── scripts/
│   ├── eda_london_crime_2024.py
│   ├── extract_theft_hotspot_polygons.py
│   ├── filter_theft_crimes.py
│   ├── theft_hotspot_stats.py
│   ├── theft_hotspots_kde.py
│   └── theft_hotspots_kde_v2.py
├── .gitignore
├── CLAUDE.md
└── README.md
```

## Main Technologies

The project primarily uses Python and the geospatial Python ecosystem.

* Pandas — tabular data processing
* GeoPandas — vector geospatial analysis
* Shapely — geometry operations
* Xarray / Rioxarray — raster and multidimensional spatial data
* Matplotlib — visualization
* Scikit-learn — machine learning
* GeoAI — deep learning and geospatial AI
* Streamlit — interactive data applications
* Leaflet — interactive web mapping

## Environment Setup

Use a Conda environment rather than installing packages into the base environment.

Create an environment:

```bash
conda create -n claude_code_geoai python=3.12
```

Activate it:

```bash
conda activate claude_code_geoai
```

Install the packages required by the scripts you intend to run.

Example:

```bash
conda install pandas geopandas matplotlib scikit-learn
```

## Running the Project

Clone the repository:

```bash
git clone https://github.com/3bdillahiomar/agentic_coding_geospatial.git
```

Enter the project:

```bash
cd agentic_coding_geospatial
```

Activate the Conda environment:

```bash
conda activate claude_code_geoai
```

Run a script, for example:

```bash
python scripts/eda_london_crime_2024.py
```

## London Crime Analysis

The London crime workflow includes scripts for:

* Exploratory data analysis
* Filtering theft-related crimes
* Kernel density estimation
* Theft hotspot extraction
* Spatial hotspot statistics
* Visualization of crime patterns

Example workflow:

```text
London crime data
      ↓
Exploratory analysis
      ↓
Filter theft crimes
      ↓
Kernel density estimation
      ↓
Extract hotspot polygons
      ↓
Spatial statistics
      ↓
Maps and analytical outputs
```

## Outputs

Generated analysis files are stored under:

```text
output/
```

Examples may include:

* PNG maps and charts
* GeoJSON files
* Hotspot polygons
* Summary statistics
* Processed spatial datasets

Generated outputs may be excluded from Git depending on the `.gitignore` configuration.

## Claude Code

The repository includes a `CLAUDE.md` file containing project-specific coding preferences, geospatial library preferences, environment rules, and development conventions for Claude Code.

Start Claude Code from the project root:

```bash
claude
```

For complex tasks, a useful workflow is:

```text
Inspect → Plan → Implement → Test → Verify
```

## Git Workflow

Check changes:

```bash
git status
```

Stage changes:

```bash
git add .
```

Commit changes:

```bash
git commit -m "Describe the changes"
```

Push changes to GitHub:

```bash
git push
```

## Repository

GitHub:

```text
https://github.com/3bdillahiomar/agentic_coding_geospatial
```
