import pandas as pd

from config import TOP_FIVE_LEAGUES
from analyze.network_analysis import filter_window

FLOW_COLUMNS = ["from", "to", "window", "value"]


def sum_flows(transfers, from_column, to_column, labels):
    """Sum the fees between every pair of labels.

    Args:
        transfers (pd.DataFrame): Transfers with the label columns and a fee column.
        from_column (str): Column with the label of the selling side.
        to_column (str): Column with the label of the buying side.
        labels (list): All labels in table order, so empty pairs get 0.

    Returns:
        pd.DataFrame: Fees with selling labels as rows and buying labels as columns.
    """
    values = transfers.groupby([from_column, to_column])["fee"].sum()
    table = values.unstack(fill_value=0)
    return table.reindex(index=labels, columns=labels, fill_value=0)


def flows_to_long_table(observed, window):
    """Turn a flow table into one row per pair.

    Args:
        observed (pd.DataFrame): Flows with selling labels as rows.
        window (str): Label stored in the window column.

    Returns:
        pd.DataFrame: One row per pair with the columns in FLOW_COLUMNS.
    """
    index = pd.MultiIndex.from_product(
        [observed.index, observed.columns], names=["from", "to"]
    )
    table = pd.DataFrame({"value": observed.to_numpy().ravel()}, index=index)
    return table.reset_index().assign(window=window)[FLOW_COLUMNS]


def build_flow_table(transfers, from_column, to_column, labels, window):
    """Build the fee flows between labels for one window.

    Args:
        transfers (pd.DataFrame): Transfers inside the window.
        from_column (str): Column with the label of the selling side.
        to_column (str): Column with the label of the buying side.
        labels (list): All labels in table order.
        window (str): Label stored in the window column.

    Returns:
        pd.DataFrame: Fee flows with the columns in FLOW_COLUMNS.
    """
    observed = sum_flows(transfers, from_column, to_column, labels)
    return flows_to_long_table(observed, window)


def build_league_flows(transfers, start_date, end_date, window):
    """Build the flows between the leagues of one window.

    Args:
        transfers (pd.DataFrame): Transfer edges with from_league, to_league and fee.
        start_date (str): First day of the window.
        end_date (str): Last day of the window.
        window (str): Label stored in the window column.

    Returns:
        pd.DataFrame: One row per league pair with the columns in FLOW_COLUMNS.
    """
    window_transfers = filter_window(transfers, start_date, end_date)
    leagues = list(TOP_FIVE_LEAGUES)
    return build_flow_table(window_transfers, "from_league", "to_league", leagues, window)
