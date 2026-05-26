# Script to make plots of vineyards from Ohio county drug induced deaths

smoothing = False
cumulative = True

# homology degree
degree = 1
# takes the top n_vines most persistent vines for plotting
n_vines = 5

#################################################################

import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from polygonal_surface import PolygonalSurface as PS

data_path = PROJECT_ROOT / "data" / "pop_normalized_county_drug_death.csv"
adj_path = PROJECT_ROOT / "data" / "ohio_neighbors.txt"


df = pd.read_csv(data_path)
# set counties to be row names
df = df.set_index('Unnamed: 0')
df.columns = np.arange(df.shape[1])

if smoothing:
    # yearly rolling avg
    df = df.T.rolling(12, min_periods=1).mean().T
if cumulative:
    df = df.cumsum(axis=1)

ps = PS.read_adj(adj_path)

# convert adj to data expected from vineyard function
vineyard_data = {}

counties = list(df.index)
for c in counties:
   vineyard_data[c] = df.loc[c].tolist()

vy = ps.toVineyard(vineyard_data, dim=degree)

top_vine_end_k = n_vines - 1

vy.print_nontrivial_vines(0, top_vine_end_k)

# Plot the top n_vines most persistent vines together with any vines that are
# represented by one of those top vines, so representative switches stay visible.
vy.plot(0, top_vine_end_k, label_by_region=True, include_represented_vines=True)
