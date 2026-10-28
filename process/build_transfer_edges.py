import pandas as pd

from process.load_data import load_club_leagues, load_clubs, load_transfers

EDGE_COLUMNS = [
    "transfer_date",
    "from_club_id",
    "to_club_id",
    "from_league",
    "to_league",
    "fee",
    "is_paid",
]

CLUB_TABLE_COLUMNS = ["club_id", "club_name", "league"]


def build_club_league_map(club_leagues):
    """Map each club to its football league.

    Args:
        club_leagues (pd.DataFrame): Rows of club_id and league_id.

    Returns:
        pd.Series: League ID indexed by club ID.
    """
    one_row_per_club = club_leagues.drop_duplicates("club_id")
    return one_row_per_club.set_index("club_id")["league_id"]


def keep_top_five_transfers(transfers, club_league_map):
    """Keep only transfers where both clubs are in the top-five leagues.

    Args:
        transfers (pd.DataFrame): Cleaned transfers.
        club_league_map (pd.Series): League ID indexed by club ID.

    Returns:
        pd.DataFrame: Transfers between two top-five league clubs.
    """
    from_known = transfers["from_club_id"].isin(club_league_map.index)
    to_known = transfers["to_club_id"].isin(club_league_map.index)
    return transfers[from_known & to_known]


def add_leagues(transfers, club_league_map):
    """Add the league of the selling and the buying club.

    Args:
        transfers (pd.DataFrame): Transfers between top-five league clubs.
        club_league_map (pd.Series): League ID indexed by club ID.

    Returns:
        pd.DataFrame: Transfers with from_league and to_league columns.
    """
    return transfers.assign(
        from_league=transfers["from_club_id"].map(club_league_map),
        to_league=transfers["to_club_id"].map(club_league_map),
    )


def add_fee(transfers):
    """Add the fee in euros, with a missing fee counted as zero.

    Args:
        transfers (pd.DataFrame): Transfers with a transfer_fee column.

    Returns:
        pd.DataFrame: Transfers with a fee column.
    """
    return transfers.assign(fee=transfers["transfer_fee"].fillna(0))


def add_is_paid(transfers):
    """Flag transfers with a fee above zero.

    Args:
        transfers (pd.DataFrame): Transfers with a fee column.

    Returns:
        pd.DataFrame: Transfers with an is_paid column.
    """
    return transfers.assign(is_paid=transfers["fee"] > 0)


def build_transfer_edges():
    """Build one row per cleaned transfer between two top-five clubs.

    Returns:
        pd.DataFrame: Transfers with the columns in EDGE_COLUMNS.
    """
    club_leagues = load_club_leagues()
    club_league_map = build_club_league_map(club_leagues)
    transfers = keep_top_five_transfers(load_transfers(), club_league_map)
    transfers = add_leagues(transfers, club_league_map)
    transfers = add_fee(transfers)
    transfers = add_is_paid(transfers)
    return transfers[EDGE_COLUMNS].reset_index(drop=True)


def build_club_table():
    """Build one row per club with its name and league.

    Returns:
        pd.DataFrame: Clubs with the columns in CLUB_TABLE_COLUMNS.
    """
    club_league_map = build_club_league_map(load_club_leagues())
    clubs = load_clubs().drop_duplicates("club_id").set_index("club_id")
    table = pd.DataFrame(
        {
            "club_name": clubs["name"].reindex(club_league_map.index),
            "league": club_league_map,
        }
    )
    return table.rename_axis("club_id").reset_index()[CLUB_TABLE_COLUMNS]
