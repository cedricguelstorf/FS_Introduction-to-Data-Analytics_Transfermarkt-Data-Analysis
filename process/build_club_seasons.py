import numpy as np
import pandas as pd

from analyze.group_clubs import build_club_groups
from analyze.network_analysis import filter_window, sum_fees_by_club
from config import SEASON_WINDOWS, WINDOWS
from process.build_transfer_edges import build_club_table, build_transfer_edges
from process.load_data import load_team_games

FIRST_SEASON = 2015
LAST_SEASON = 2025
EXPECTED_TEAMS = {18, 20}
EXPECTED_GAMES = {34, 38}
WIN_POINTS = 3
DRAW_POINTS = 1
MILLION = 1e6
CLUB_SEASON_COLUMNS = [
    "club_id",
    "season",
    "league",
    "group",
    "ppg",
    "previous_ppg",
    "fees_paid_mio",
    "fees_received_mio",
    "net_spend_mio",
]


def add_points(team_games):
    """Add the points each team won in each game.

    Args:
        team_games (pd.DataFrame): Output of load_team_games.

    Returns:
        pd.DataFrame: Team games with a points column.
    """
    is_win = team_games["goals_for"] > team_games["goals_against"]
    is_draw = team_games["goals_for"] == team_games["goals_against"]
    points = np.select([is_win, is_draw], [WIN_POINTS, DRAW_POINTS], 0)
    return team_games.assign(points=points)


def sum_season_results(team_games):
    """Sum points and goals per club and season and add points per game.

    Args:
        team_games (pd.DataFrame): Team games with a points column.

    Returns:
        pd.DataFrame: One row per club and season with games, points, goals_for,
            goals_against and ppg.
    """
    results = team_games.groupby(["club_id", "season", "league"]).agg(
        games=("points", "size"),
        points=("points", "sum"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
    )
    results = results.reset_index()
    return results.assign(ppg=results["points"] / results["games"])


def add_table_position(results):
    """Add the final table position of each club in its league and season.

    Clubs are ranked by points, then goal difference, then goals scored.

    Args:
        results (pd.DataFrame): Output of sum_season_results.

    Returns:
        pd.DataFrame: Results with a table_position column.
    """
    goal_diff = results["goals_for"] - results["goals_against"]
    ordered = results.assign(goal_diff=goal_diff).sort_values(
        ["league", "season", "points", "goal_diff", "goals_for"],
        ascending=[True, True, False, False, False],
    )
    positions = ordered.groupby(["league", "season"]).cumcount() + 1
    return results.assign(table_position=positions)


def add_previous_ppg(results):
    """Add the points per game of the season before.

    Args:
        results (pd.DataFrame): Results with club_id, season and ppg.

    Returns:
        pd.DataFrame: Results with a previous_ppg column, empty if the club did not
            play a top-five league in the season before.
    """
    previous = results[["club_id", "season", "ppg"]].rename(columns={"ppg": "previous_ppg"})
    previous = previous.assign(season=previous["season"] + 1)
    return results.merge(previous, on=["club_id", "season"], how="left")


def build_performance(team_games):
    """Build the season results of every club.

    Args:
        team_games (pd.DataFrame): Output of load_team_games.

    Returns:
        pd.DataFrame: One row per club and season with games, points, goals, ppg,
            table_position and previous_ppg.
    """
    results = sum_season_results(add_points(team_games))
    return add_previous_ppg(add_table_position(results))


def report_unusual_league_seasons(performance):
    """Print the league seasons that do not have 18 or 20 teams with 34 or 38 games each.

    Args:
        performance (pd.DataFrame): Output of build_performance.
    """
    checks = performance.groupby(["league", "season"]).agg(
        teams=("club_id", "size"),
        min_games=("games", "min"),
        max_games=("games", "max"),
    )
    is_usual = (
        checks["teams"].isin(EXPECTED_TEAMS)
        & checks["max_games"].isin(EXPECTED_GAMES)
        & (checks["min_games"] == checks["max_games"])
    )
    print(f"League seasons: {len(checks)}, unusual: {(~is_usual).sum()}")
    if (~is_usual).any():
        print(checks[~is_usual].to_string())


def build_transfer_fees(transfers, club_ids):
    """Sum the fees each club paid and received in each season.

    Args:
        transfers (pd.DataFrame): Transfer edges with fee and club ID columns.
        club_ids (pd.Index): All club IDs, so clubs without transfers get 0.

    Returns:
        pd.DataFrame: One row per club and season with fees_paid_mio,
            fees_received_mio and net_spend_mio in million euros.
    """
    tables = []
    for season, (start_date, end_date) in SEASON_WINDOWS.items():
        season_transfers = filter_window(transfers, start_date, end_date)
        paid = sum_fees_by_club(season_transfers, "to_club_id", club_ids)
        received = sum_fees_by_club(season_transfers, "from_club_id", club_ids)
        table = pd.DataFrame(
            {
                "season": int(season[:4]),
                "fees_paid_mio": paid / MILLION,
                "fees_received_mio": received / MILLION,
                "net_spend_mio": (paid - received) / MILLION,
            }
        )
        tables.append(table.reset_index())
    return pd.concat(tables, ignore_index=True)


def build_club_seasons():
    """Join results, transfer fees and club group for every club season.

    Only club seasons with transfer data and a top-five season before are kept.

    Returns:
        pd.DataFrame: One row per club and season with the columns in CLUB_SEASON_COLUMNS.
    """
    performance = build_performance(load_team_games(FIRST_SEASON, LAST_SEASON))
    report_unusual_league_seasons(performance)
    transfers = build_transfer_edges()
    club_table = build_club_table()
    club_ids = pd.Index(club_table["club_id"], name="club_id")
    fees = build_transfer_fees(transfers, club_ids)
    groups = build_club_groups(transfers, club_table, *WINDOWS["full"], "full")
    data = performance.merge(fees, on=["club_id", "season"], how="inner")
    data = data[data["previous_ppg"].notna()]
    data = data.merge(groups[["club_id", "group"]], on="club_id", how="left")
    data = data.sort_values(["season", "league", "table_position"]).reset_index(drop=True)
    return data[CLUB_SEASON_COLUMNS]
