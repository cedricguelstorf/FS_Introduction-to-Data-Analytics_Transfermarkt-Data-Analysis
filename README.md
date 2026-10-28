# Transfermarkt Data Analysis: Top 5 Transfer Networks

In this project, we analyze the network of player transfers between clubs of the five top European leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1), from July 2016 to July 2026.

## Setup

To prepare your setup for this project run this command in your terminal:

pip install -r requirements.txt

## Data

The data comes from a Transfermarkt dataset found on Kaggle: https://www.kaggle.com/datasets/davidcariboo/player-scores/data

The files the code needs are saved in the folder `data` of this repository. These are the only files of that dataset that the code reads: "transfer.csv", "games.csv", and "clubs.csv".

The folder also holds `club_locations.csv`, the stadium coordinates of the clubs. It is used only to place the clubs on the map of the Gephi network.

The analysis covers 1 July 2016 to 6 July 2026, which are the seasons 2016/17 to 2025/26. All other files of the Kaggle dataset are not used and are not in the folder (some files exceed 100MB).

## Folder overview

The code follows four steps. Each step has its own folder.

1. `process/` prepares the data.
   - `load_data.py` reads and cleans the files in `data`.
   - `build_transfer_edges.py` builds one row per transfer between two top-five clubs and a table of the clubs with their league.
   - `build_club_seasons.py` builds one row per club and season with points per game, previous points per game, fees and club group.
2. `analyze/` computes the results.
   - `network_analysis.py` and `network_structure.py` build the club network, the trading partners of each club and the random reference networks.
   - `group_clubs.py` sorts the clubs into giving, taking, two-way and low-activity clubs
   - `build_flows.py` and `change_over_time.py` compute the fees between the leagues, in total and per season.
   - `club_performance.py` summarises points per game by group and fits the line of points on net spend.
   - `model_comparison.py` compares a baseline with a linear regression and a random forest on a time-based split.
3. `use/` makes the figures and the network files.
   - `plot_figures.py`, `plot_random_band.py` and `plot_performance.py` hold the plot functions.
   - `create_figures.py` creates all figures.
   - `build_gephi_network.py` creates the node and edge files for Gephi.
4. `output/` contains the figures as PNG files, `nodes.csv` and `edges.csv` for Gephi, and `model_comparison.csv`.

`config.py` holds the settings, colours and paths.