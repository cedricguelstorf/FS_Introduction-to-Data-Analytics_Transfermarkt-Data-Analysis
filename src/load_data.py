import pandas as pd

from config import DATA_DIR, END_DATE, PSEUDO_CLUB_IDS, START_DATE, TOP_FIVE_LEAGUES

RAW_TRANSFER_COLUMNS = [
    "player_id",
    "transfer_date",
    "transfer_season",
    "from_club_id",
    "to_club_id",
    "from_club_name",
    "to_club_name",
    "transfer_fee",
    "market_value_in_eur",
]

TRANSFER_COLUMNS = [
    "player_id",
    "transfer_date",
    "season",
    "from_club_id",
    "to_club_id",
    "from_club_name",
    "to_club_name",
    "transfer_fee",
    "market_value_in_eur",
]


def season_to_year(season):
    """Convert a season label such as "25/26" to its start year 2025."""
    return 2000 + int(season.split("/")[0])


def read_transfers():
    """Read the needed columns of transfers.csv without cleaning."""
    return pd.read_csv(
        DATA_DIR / "transfers.csv",
        usecols=RAW_TRANSFER_COLUMNS,
        parse_dates=["transfer_date"],
    )


def filter_dates(transfers):
    """Keep transfers from START_DATE to END_DATE inclusive."""
    in_window = transfers["transfer_date"].between(START_DATE, END_DATE)
    return transfers[in_window]


def drop_self_loops(transfers):
    """Drop transfers where the club IDs are equal."""
    return transfers[transfers["from_club_id"] != transfers["to_club_id"]]


def drop_pseudo_clubs(transfers):
    """Drop transfers from or to a pseudo-club."""
    from_pseudo = transfers["from_club_id"].isin(PSEUDO_CLUB_IDS)
    to_pseudo = transfers["to_club_id"].isin(PSEUDO_CLUB_IDS)
    return transfers[~(from_pseudo | to_pseudo)]


def load_transfers():
    """Load cleaned transfers between real clubs, with an integer season."""
    transfers = read_transfers()
    transfers = filter_dates(transfers)
    transfers = drop_self_loops(transfers)
    transfers = drop_pseudo_clubs(transfers)
    transfers = transfers.assign(season=transfers["transfer_season"].map(season_to_year))
    return transfers[TRANSFER_COLUMNS].reset_index(drop=True)


def load_club_leagues():
    """Load the top-five league of each club in each season.

    Returns:
        pd.DataFrame: Unique rows of club_id, season and league_id.
    """
    games = pd.read_csv(
        DATA_DIR / "games.csv",
        usecols=["competition_id", "season", "home_club_id", "away_club_id"],
    )
    games = games[games["competition_id"].isin(TOP_FIVE_LEAGUES)]
    home = games.rename(columns={"home_club_id": "club_id"})
    away = games.rename(columns={"away_club_id": "club_id"})
    columns = ["club_id", "season", "competition_id"]
    stacked = pd.concat([home[columns], away[columns]])
    stacked = stacked.rename(columns={"competition_id": "league_id"})
    return stacked.drop_duplicates().reset_index(drop=True)


def load_clubs():
    """Load club IDs and names."""
    return pd.read_csv(DATA_DIR / "clubs.csv", usecols=["club_id", "name"])
