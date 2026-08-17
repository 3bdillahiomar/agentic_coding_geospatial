# Agentic Coding for Geospatial

A Python-based geospatial project for exploring spatial data workflows, automation, visualization, machine learning, and GeoAI-assisted development.

The repository includes examples for London crime analysis, theft hotspot detection, spatial statistics, raster and vector workflows, and geospatial visualization.

![Project cover](figures/git_cover.png)

## Project Structure

```text
agentic_coding_geospatial/
├── data/
│   ├── chirps/
│   ├── london_crime_2024/
│   └── route_optimization/
├── documents/
├── figures/
│   └── git_cover.png
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
* Matplotlib — plotting and visualization
* Scikit-learn — machine learning
* GeoAI — deep learning and geospatial AI
* Streamlit — interactive data applications
* Leaflet — interactive web mapping

## Environment Setup

Use a Conda environment rather than installing packages into the base environment.

Create the environment:

```bash
conda create -n claude_code_geoai python=3.12
```

Activate it:

```bash
conda activate claude_code_geoai
```

Install the main packages:

```bash
conda install pandas geopandas matplotlib scikit-learn
```

Install additional geospatial packages as required by individual workflows.

## Running the Project

Clone the repository:

```bash
git clone https://github.com/3bdillahiomar/agentic_coding_geospatial.git
```

Move into the project directory:

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

The London crime workflow includes:

* Exploratory data analysis
* Filtering theft-related crimes
* Kernel density estimation
* Theft hotspot extraction
* Spatial hotspot statistics
* Geospatial visualization

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

Outputs may include:

* PNG maps and charts
* GeoJSON files
* Hotspot polygons
* Summary statistics
* Processed spatial datasets

Generated outputs may be excluded from Git depending on the `.gitignore` configuration.

## Claude Code

The repository includes a `CLAUDE.md` file containing project-specific instructions for Claude Code, including:

* Preferred geospatial libraries
* Conda environment rules
* Coding preferences
* Development conventions

Start Claude Code from the project root:

```bash
claude
```

For complex tasks, use the following workflow:

```text
Inspect → Plan → Implement → Test → Verify
```

## Git Workflow

Check the current repository status:

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
