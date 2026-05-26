# Ohio Vineyards

Ohio-focused copy of the vineyard analysis code for county-level overdose data.

This repository contains:

- Core vineyard and polygonal-surface modules
- Ohio county overdose datasets
- Ohio county adjacency data
- Scripts for vineyard plots and top-county time-series plots
- Example generated figures

## Repository Layout

- `vineyard.py`: vineyard algorithm implementation
- `polygonal_surface.py`: polygonal surface and adjacency handling
- `scripts/ohio_vineyards.py`: builds and plots Ohio vineyards
- `scripts/ohio_vineyard_hypothesis_test.py`: county-label scrambling hypothesis test for vineyard statistics
- `scripts/deathcount_plots.py`: standard top-county line plots
- `data/`: overdose datasets and Ohio county adjacency file
- `figures/`: generated example outputs


## Usage

Generate the Ohio vineyard plot:

```bash
python scripts/ohio_vineyards.py
```

Generate the standard top-county death-count plots:

```bash
python scripts/deathcount_plots.py
```

Run the county-label scrambling hypothesis test:

```bash
python scripts/ohio_vineyard_hypothesis_test.py --trials 100 --stat all --k 5
```

The test keeps the Ohio county adjacency map fixed, randomly reassigns whole county time series to county labels, rebuilds the vineyard for each scramble, and compares the observed vineyard statistic to that null distribution. This tests whether the observed placement of county histories on the Ohio map is unusual relative to random geographic reassignment. By default, the null distribution is saved to `data/null_distributions/ohio_vineyard_null_distribution.csv`.

Available test statistics:

- `top_persistence`: persistence of the single most persistent finite vine.
- `top_k_mean`: average persistence among the top `k` finite vines.
- `top_k_sum`: total persistence among the top `k` finite vines.
- `total_persistence`: total persistence across all finite nontrivial vines.
- `top_persistence_share`: fraction of total finite persistence carried by the top vine.
- `finite_vine_count`: number of finite nontrivial vines.
- `persistence_entropy`: entropy of the finite persistence distribution; larger values mean persistence is more evenly spread across vines.

Use `--stat top_persistence`, `--stat top_k_mean --k 5`, or `--stat all`. Use `--list-stats` to print the available statistics from the script.

## Data

The repository currently includes:

- `data/county_drug_death.csv`
- `data/pop_normalized_county_drug_death.csv`
- `data/ohio_neighbors.txt`
