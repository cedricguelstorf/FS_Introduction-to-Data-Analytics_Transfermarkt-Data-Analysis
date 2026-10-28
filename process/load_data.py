import pandas as pd

from config import (
    CLUBS_PATH,
    END_DATE,
    GAMES_PATH,
    PSEUDO_CLUB_IDS,
    START_DATE,
    TOP_FIVE_LEAGUES,
    TRANSFERS_PATH,
)

TRANSFER_COLUMNS = [
    "transfer_date",
    "from_club_id",
    "to_club_id",
    "transfer_fee",
]

GAME_COLUMNS = [
    "competition_id",
    "season",
    "home_club_id",
    "away_club_id",
    "home_club_goals",
    "away_club_goals",
]

TEAM_GAME_COLUMNS = ["club_id", "season", "league", "goals_for", "goals_against"]


def read_transfers():
    """Read the needed columns of transfers.csv without cleaning.

    Returns:
        pd.DataFrame: Raw transfers with the columns in TRANSFER_COLUMNS.
    """
    return pd.read_csv(
        TRANSFERS_PATH,
        usecols=TRANSFER_COLUMNS,
        parse_dates=["transfer_date"],
    )


def filter_dates(transfers):
    """Keep transfers from START_DATE to END_DATE inclusive.

    Args:
        transfers (pd.DataFrame): Transfers with a transfer_date column.

    Returns:
        pd.DataFrame: Transfers inside the date window.
    """
    in_window = transfers["transfer_date"].between(START_DATE, END_DATE)
    return transfers[in_window]


def drop_self_loops(transfers):
    """Drop transfers where the club IDs are equal.

    Args:
        transfers (pd.DataFrame): Transfers with from_club_id and to_club_id.

    Returns:
        pd.DataFrame: Transfers between two different clubs.
    """
    return transfers[transfers["from_club_id"] != transfers["to_club_id"]]


def drop_pseudo_clubs(transfers):
    """Drop transfers from or to a pseudo-club.

    Args:
        transfers (pd.DataFrame): Transfers with from_club_id and to_club_id.

    Returns:
        pd.DataFrame: Transfers between real clubs.
    """
    from_pseudo = transfers["from_club_id"].isin(PSEUDO_CLUB_IDS)
    to_pseudo = transfers["to_club_id"].isin(PSEUDO_CLUB_IDS)
    return transfers[~(from_pseudo | to_pseudo)]


def load_transfers():
    """Load cleaned transfers between real clubs.

    Returns:
        pd.DataFrame: Transfers with the columns in TRANSFER_COLUMNS.
    """
    transfers = read_transfers()
    transfers = filter_dates(transfers)
    transfers = drop_self_loops(transfers)
    transfers = drop_pseudo_clubs(transfers)
    return transfers.reset_index(drop=True)


def load_club_leagues():
    """Load the top-five league of each club.

    Returns:
        pd.DataFrame: Unique rows of club_id and league_id.
    """
    games = pd.read_csv(
        GAMES_PATH,
        usecols=["competition_id", "home_club_id", "away_club_id"],
    )
    games = games[games["competition_id"].isin(TOP_FIVE_LEAGUES)]
    home = games.rename(columns={"home_club_id": "club_id"})
    away = games.rename(columns={"away_club_id": "club_id"})
    columns = ["club_id", "competition_id"]
    stacked = pd.concat([home[columns], away[columns]])
    stacked = stacked.rename(columns={"competition_id": "league_id"})
    return stacked.drop_duplicates().reset_index(drop=True)


def load_clubs():
    """Load club IDs and names.

    Returns:
        pd.DataFrame: One row per club with club_id and name.
    """
    return pd.read_csv(CLUBS_PATH, usecols=["club_id", "name"])


def build_team_rows(games, side, other):
    """Turn each game into the row of one side.

    Args:
        games (pd.DataFrame): Games with the columns in GAME_COLUMNS.
        side (str): "home" or "away", the side the row belongs to.
        other (str): The other side, "away" or "home".

    Returns:
        pd.DataFrame: One row per game with the columns in TEAM_GAME_COLUMNS.
    """
    rows = games.rename(
        columns={
            "competition_id": "league",
            f"{side}_club_id": "club_id",
            f"{side}_club_goals": "goals_for",
            f"{other}_club_goals": "goals_against",
        }
    )
    return rows[TEAM_GAME_COLUMNS]


def load_team_games(first_season, last_season):
    """Load the top-five league games with one row per team and game.

    Args:
        first_season (int): First season to keep, given by its start year.
        last_season (int): Last season to keep, given by its start year.

    Returns:
        pd.DataFrame: Rows with the columns in TEAM_GAME_COLUMNS.
    """
    games = pd.read_csv(GAMES_PATH, usecols=GAME_COLUMNS)
    is_top_five = games["competition_id"].isin(TOP_FIVE_LEAGUES)
    in_seasons = games["season"].between(first_season, last_season)
    games = games[is_top_five & in_seasons]
    home = build_team_rows(games, "home", "away")
    away = build_team_rows(games, "away", "home")
    return pd.concat([home, away], ignore_index=True)
