from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

TOP_FIVE_LEAGUES = {
    "GB1": "Premier League",
    "ES1": "La Liga",
    "IT1": "Serie A",
    "L1": "Bundesliga",
    "FR1": "Ligue 1",
}

START_DATE = "2016-07-01"
END_DATE = "2026-07-06"

WINDOWS = {
    "full": (START_DATE, END_DATE),
}

SEASON_WINDOWS = {
    f"{year}_{str(year + 1)[2:]}": (f"{year}-07-01", f"{year + 1}-06-30")
    for year in range(2016, 2026)
}

PSEUDO_CLUB_IDS = {75, 123, 515, 2113}

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
FEES_FIGSIZE = (20.07 / 2.54, 11.43 / 2.54)
NET_BALANCE_FIGSIZE = (14.9 / 2.54, 11.43 / 2.54)
DEGREE_FIGSIZE = (20.07 / 2.54, 11.43 / 2.54)
HEATMAP_SLIDE_FIGSIZE = (14.9 / 2.54, 11.43 / 2.54)
BUY_SELL_FIGSIZE = (20.07 / 2.54, 11.43 / 2.54)
PERFORMANCE_FIGSIZE = (14.9 / 2.54, 11.43 / 2.54)

LEAGUE_COLORS = {
    "GB1": "#7f46b5",
    "ES1": "#e5761a",
    "IT1": "#2189da",
    "L1": "#cf2030",
    "FR1": "#6fa300",
}
GROUP_MARKERS = {"giving": "^", "taking": "v", "two_way": "o", "low_activity": "o"}
GROUP_LABELS = {
    "giving": "Giving",
    "two_way": "Two-way",
    "taking": "Taking",
    "low_activity": "Low activity",
}
GROUP_COLORS = {
    "taking": "#3e6e7e",
    "two_way": "#b9b2a3",
    "giving": "#8f4a46",
    "low_activity": "#d4d3cd",
}

SURFACE_COLOR = "#fcfcfb"
INK_COLOR = "#0b0b0b"
SECONDARY_INK_COLOR = "#52514e"
MUTED_COLOR = "#898781"
GRID_COLOR = "#e1e0d9"
OTHER_LEAGUES_COLOR = "#d4d3cd"

MIN_PAID_TRANSFERS = 10
N_FLIPS = 1000
BAND_SHARE = 0.95
RANDOM_SEED = 42
N_REFERENCE_NETWORKS = 100

TRANSFERS_PATH = DATA_DIR / "transfers.csv"
GAMES_PATH = DATA_DIR / "games.csv"
CLUBS_PATH = DATA_DIR / "clubs.csv"
CLUB_LOCATIONS_PATH = DATA_DIR / "club_locations.csv"
MODEL_COMPARISON_PATH = OUTPUT_DIR / "model_comparison.csv"

MIN_CLUB_DISTANCE_KM = 45
MAX_SPREAD_ROUNDS = 300
KM_PER_DEGREE_LATITUDE = 110.57
KM_PER_DEGREE_LONGITUDE_AT_EQUATOR = 111.32
REFERENCE_LATITUDE = 46
LAS_PALMAS_CLUB_ID = 472
LAS_PALMAS_DISPLAY_POSITION = (34.0, -10.8)
COUNTRY_BOUNDS = {
    "GB1": (49.8, 60.9, -8.7, 1.8),
    "ES1": (27.5, 43.9, -18.3, 4.4),
    "IT1": (36.5, 47.2, 6.5, 18.6),
    "L1": (47.2, 55.1, 5.8, 15.1),
    "FR1": (41.3, 51.2, -5.2, 9.7),
}

N_LABELLED_CLUBS = 15
