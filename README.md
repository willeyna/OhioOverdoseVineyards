# Characterizing Local Maxima of the Ohio overdose epidemic with Vineyards 

This repository contains the code, data, and notebooks to analyze the spatial and temporal patterns of the Ohio drug overdose epidemic using topological data analysis (TDA)—specifically the **Vineyard** algorithm—along with classical statistical methods.

## Directory Structure & Contents

- `vineyard.py`: Vineyard algorithm implementation (originally from [A. Hickok's repository](https://bitbucket.org/ahickok/vineyard/src/main/scripts/)).
- `polygonal_surface.py`: Helper for the Vineyard algorithm on county adjacencies.
- `data/`: Contains Ohio drug overdose deaths from 2007–2024 (raw counts and normalized rates) and county adjacency mappings.
- `figures/`: Generated vineyard plots and other supplementary diagrams.
- `ohio_vineyard_scripts/`: 
  - `analyze_ohio.ipynb`: Notebook analyzing overdose vineyards.
  - `ohio_vineyards.py`: Script to generate vineyards individually.

### Sub-studies

The analysis is broken down into four main sub-study directories. Notebooks in these directories run dynamically against the shared root `data/` folder and `vineyard` package:

1. **`uniform_hypotheses/`**: Testing uniform null hypothesis models for Ohio county-month overdose deaths.
   - `uniform-testing.qmd`: Quarto notebook running uniformity and spatial-autocorrelation tests.
   - `uniform-testing.html`: Pre-rendered HTML report.
   - `uniform_testing_outputs/`: Saved CSV outputs (simulated null replicates, monthwise p-values, etc.).
   
2. **`forecasting/`**: Time-series forecasting and evaluation of Ohio overdose rates and persistence features.
   - `forecasting-vineyards.qmd`: Quarto notebook detailing the forecasting models (SARIMAX, etc.).
   - `forecasting-vineyards.html`: Pre-rendered HTML report.
   - `forecasting_vineyards_outputs/`: Saved CSV outputs (forecast holdouts, significance metrics, etc.) and selected plots.

3. **`spatiotemporal_correlation/`**: Analysis of spatiotemporal autocorrelation using classical covariance and TDA tests.
   - `spatiotemporal-autocorrelation-classical-and-tda.qmd`: Quarto notebook for spatiotemporal autocorrelation.
   - `spatiotemporal-autocorrelation-classical-and-tda.html`: Pre-rendered HTML report.
   - `spatiotemporal_autocorrelation_outputs/`: Saved output files (prepared spatiotemporal panel data, covariance matrices, and TDA summaries).

4. **`confidence_intervals/`**: Constructing confidence bands and intervals for vineyard persistence diagrams and metrics.
   - `confidence-intervals-vineyards.qmd`: Quarto notebook for confidence interval estimation.
   - `confidence-intervals-vineyards.html`: Pre-rendered HTML report.
   - `confidence_intervals_outputs/`: Saved CSV outputs containing simulated null replicates, monthwise vineyard tube geometries, and other output from the TDA computations.

---

## Usage

### Creating Vineyards Individually
You can run the script to generate individual vineyards. You can configure parameters inside the script to choose the data file (raw count or normalized) and the number of vines to plot:
```bash
python ohio_vineyard_scripts/ohio_vineyards.py
```

### Rendering Study Notebooks
To render and execute any of the sub-study notebooks via Quarto, run the `quarto render` command pointing to the respective `.qmd` file:
```bash
# E.g. to render the uniformity study notebook:
quarto render "uniform_hypotheses/uniform-testing.qmd"
```

> [!NOTE]
> The sub-study notebooks are written so that they can be rendered from their respective directories without using sibling folders from the larger project. The computationally expensive source cells that create the saved output files are included for reproducibility but are not executed during ordinary rendering (they are marked `eval: false`) so that the documents compile quickly using the cached results.
