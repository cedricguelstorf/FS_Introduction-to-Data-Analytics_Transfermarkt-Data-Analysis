import pandas as pd

from analyze.build_flows import build_league_flows
from config import SEASON_WINDOWS


def build_season_league_flows(transfers):
    """Build the league flows of every season.

    Args:
        transfers (pd.DataFrame): Transfer edges with from_league, to_league and fee.

    Returns:
        pd.DataFrame: Output of build_league_flows for all seasons stacked.
    """
    tables = [
        build_league_flows(transfers, start_date, end_date, season)
        for season, (start_date, end_date) in SEASON_WINDOWS.items()
    ]
    return pd.concat(tables, ignore_index=True)
