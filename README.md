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

## Requirements

Tested with Python 3.10+.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

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

If you publish this repository, make sure you are comfortable sharing these files under your intended distribution terms.

## Notes

- The vineyard plot script currently uses cumulative counts by default.
- The plotting logic now includes the top `k` seed vines together with any vine that shares a representative county with one of those seed vines.
- Notebook files are included for analysis, but the runnable entry points for others are the Python scripts above.

## GitHub

To publish after Git initialization:

```bash
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

This repository does not yet include a license file. If you want others to reuse it clearly, add a license before publishing.
