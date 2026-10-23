from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"

TOP_FIVE_LEAGUES = {
    "GB1": "Premier League",
    "ES1": "La Liga",
    "IT1": "Serie A",
    "L1": "Bundesliga",
    "FR1": "Ligue 1",
}

START_DATE = "2013-07-01"
END_DATE = "2026-07-06"

PSEUDO_CLUB_IDS = {75, 123, 515, 2113}
