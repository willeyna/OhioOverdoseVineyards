# Characterizing Local Maxima of the Ohio overdose epidemic with Vineyards 

## Contents

- `vineyard.py`: Vineyard algorithm implementation, originally from https://bitbucket.org/ahickok/vineyard/src/main/scripts/  
- `polygonal_surface.py`: Helper for the Vineyard algorithm on county adjacencies 
- `scripts/analyze_ohio.ipynb`: Notebook containing vineyards of our data
- `scripts/ohio_vineyards.py`: Script to generate vineyards individually 
- `data/`: Contains ohio drug overdose deaths from 2007-2024 (raw count and normalized by county population) as well as adjacency relations between the counties. 
- `figures/`: Vineyard results and other supplementary figures 

## Usage

### Creating Vineyards 

The following can be used to create Vineyards. The script contains parameters for choosing the data file (raw count or normalized) as well as the number of vines to plot 
```bash
python scripts/ohio_vineyards.py
```
