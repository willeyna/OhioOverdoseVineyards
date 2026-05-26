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

## Data

The repository currently includes:

- `data/county_drug_death.csv`
- `data/pop_normalized_county_drug_death.csv`
- `data/ohio_neighbors.txt`


