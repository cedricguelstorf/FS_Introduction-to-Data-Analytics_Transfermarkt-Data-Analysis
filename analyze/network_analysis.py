import networkx as nx
import pandas as pd

MEASURE_COLUMNS = ["club_id", "fees_received", "fees_paid", "partners"]


def count_club_edges(transfers):
    """Count the transfers for each pair of selling and buying club.

    Args:
        transfers (pd.DataFrame): Transfers with from_club_id and to_club_id.

    Returns:
        pd.DataFrame: One row per club pair with from_club_id, to_club_id and weight.
    """
    pairs = transfers.groupby(["from_club_id", "to_club_id"]).size()
    return pairs.rename("weight").reset_index()


def build_club_graph(transfers, club_league_map, club_names):
    """Build the directed club network, weighted by the number of transfers.

    Args:
        transfers (pd.DataFrame): Transfers between top-five league clubs.
        club_league_map (pd.Series): League ID indexed by club ID.
        club_names (pd.Series): Club name indexed by club ID.

    Returns:
        nx.DiGraph: One node per club with league and name, one edge per club pair.
    """
    graph = nx.DiGraph()
    for club_id, league in club_league_map.items():
        graph.add_node(club_id, league=league, name=club_names[club_id])
    edges = count_club_edges(transfers)
    graph.add_weighted_edges_from(edges.itertuples(index=False, name=None))
    return graph


def filter_window(transfers, start_date, end_date):
    """Keep transfers from start_date to end_date inclusive.

    Args:
        transfers (pd.DataFrame): Transfers with a transfer_date column.
        start_date (str): First day of the window, for example "2016-07-01".
        end_date (str): Last day of the window.

    Returns:
        pd.DataFrame: Transfers inside the window.
    """
    in_window = transfers["transfer_date"].between(start_date, end_date)
    return transfers[in_window]


def count_by_club(transfers, club_column, club_ids):
    """Count the transfers of each club in one role.

    Args:
        transfers (pd.DataFrame): Transfers with the club column.
        club_column (str): from_club_id to count sales, to_club_id to count purchases.
        club_ids (pd.Index): All club IDs, so clubs without transfers get 0.

    Returns:
        pd.Series: Transfer count indexed by club ID.
    """
    counts = transfers.groupby(club_column).size()
    return counts.reindex(club_ids, fill_value=0)


def sum_fees_by_club(transfers, club_column, club_ids):
    """Sum the fees of each club in one role.

    Args:
        transfers (pd.DataFrame): Transfers with a fee column.
        club_column (str): from_club_id for fees received, to_club_id for fees paid.
        club_ids (pd.Index): All club IDs, so clubs without transfers get 0.

    Returns:
        pd.Series: Fee sum in euros indexed by club ID.
    """
    fees = transfers.groupby(club_column)["fee"].sum()
    return fees.reindex(club_ids, fill_value=0)


def compute_balance(fees_received, fees_paid):
    """Compute the share of fee volume that a club received instead of paid.

    Args:
        fees_received (pd.Series): Fees received per club.
        fees_paid (pd.Series): Fees paid per club.

    Returns:
        pd.Series: Balance from -1 (only buys) to +1 (only sells), NaN without fees.
    """
    volume = fees_received + fees_paid
    return (fees_received - fees_paid) / volume.where(volume > 0)


def build_fee_measures(transfers, club_ids):
    """Build the fees each club received and paid.

    Args:
        transfers (pd.DataFrame): Transfers inside one window.
        club_ids (pd.Index): All club IDs.

    Returns:
        pd.DataFrame: Columns fees_received and fees_paid, indexed by club ID.
    """
    return pd.DataFrame(
        {
            "fees_received": sum_fees_by_club(transfers, "from_club_id", club_ids),
            "fees_paid": sum_fees_by_club(transfers, "to_club_id", club_ids),
        }
    )


def build_graph_measures(transfers, club_table):
    """Build the number of trading partners of each club.

    Partners are counted in the graph without edge directions, ignoring edge weights.

    Args:
        transfers (pd.DataFrame): Transfers inside one window.
        club_table (pd.DataFrame): Rows of club_id, club_name and league.

    Returns:
        pd.DataFrame: Column partners, indexed by club ID.
    """
    clubs = club_table.set_index("club_id")
    graph = build_club_graph(transfers, clubs["league"], clubs["club_name"])
    undirected = graph.to_undirected()
    return pd.DataFrame({"partners": pd.Series(dict(undirected.degree()))})


def build_club_network_measures(transfers, club_table, start_date, end_date):
    """Build the measures of every club for one window.

    Args:
        transfers (pd.DataFrame): Transfer edges with fee and league columns.
        club_table (pd.DataFrame): Rows of club_id, club_name and league.
        start_date (str): First day of the window.
        end_date (str): Last day of the window.

    Returns:
        pd.DataFrame: One row per club with the columns in MEASURE_COLUMNS.
    """
    window_transfers = filter_window(transfers, start_date, end_date)
    club_ids = pd.Index(club_table["club_id"], name="club_id")
    measures = build_fee_measures(window_transfers, club_ids)
    measures = measures.join(build_graph_measures(window_transfers, club_table))
    return measures.reset_index()[MEASURE_COLUMNS]
