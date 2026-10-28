import numpy as np
import pandas as pd

from config import BAND_SHARE, MIN_PAID_TRANSFERS, N_FLIPS, RANDOM_SEED
from analyze.network_analysis import (
    compute_balance,
    count_by_club,
    filter_window,
    sum_fees_by_club,
)

GROUP_COLUMNS = [
    "club_id",
    "window",
    "group",
    "balance",
    "band_low",
    "band_high",
    "paid_transfers",
]


def build_signed_fees(paid, club_ids):
    """Build the fee of each paid transfer, signed by the side of each club.

    A club's entry is positive when it sells and negative when it buys.

    Args:
        paid (pd.DataFrame): Paid transfers with from_club_id, to_club_id and fee.
        club_ids (pd.Index): All club IDs, one table column each.

    Returns:
        np.ndarray: Fees with one row per transfer and one column per club.
    """
    signed_fees = np.zeros((len(paid), len(club_ids)))
    rows = np.arange(len(paid))
    fees = paid["fee"].to_numpy()
    signed_fees[rows, club_ids.get_indexer(paid["from_club_id"])] = fees
    signed_fees[rows, club_ids.get_indexer(paid["to_club_id"])] = -fees
    return signed_fees


def draw_directions(n_flips, n_transfers, seed):
    """Draw random transfer directions, one row per repetition.

    Args:
        n_flips (int): Number of repetitions.
        n_transfers (int): Number of paid transfers.
        seed (int): Seed of the random number generator.

    Returns:
        np.ndarray: -1 where seller and buyer swap places, +1 where they do not.
    """
    rng = np.random.default_rng(seed)
    return 1 - 2 * rng.integers(0, 2, size=(n_flips, n_transfers))


def simulate_balances(signed_fees, directions):
    """Compute every club's balance in every random repetition.

    Args:
        signed_fees (np.ndarray): Output of build_signed_fees.
        directions (np.ndarray): Output of draw_directions.

    Returns:
        np.ndarray: Balances with one row per repetition and one column per club,
            NaN for clubs without fees.
    """
    net = directions @ signed_fees
    volume = np.abs(signed_fees).sum(axis=0)
    return np.divide(net, volume, out=np.full(net.shape, np.nan), where=volume > 0)


def compute_band(simulated, club_ids, band_share):
    """Compute the middle share of each club's random balances.

    Args:
        simulated (np.ndarray): Output of simulate_balances.
        club_ids (pd.Index): All club IDs, one per column of simulated.
        band_share (float): Share of the random balances inside the band, e.g. 0.95.

    Returns:
        pd.DataFrame: Columns band_low and band_high, indexed by club ID.
    """
    tail = (1 - band_share) / 2 * 100
    low, high = np.percentile(simulated, [tail, 100 - tail], axis=0)
    return pd.DataFrame({"band_low": low, "band_high": high}, index=club_ids)


def assign_groups(measures, min_paid_transfers):
    """Assign each club to a group by comparing its balance with its band.

    Args:
        measures (pd.DataFrame): Columns balance, band_low, band_high and paid_transfers.
        min_paid_transfers (int): Clubs with fewer paid transfers are low activity.

    Returns:
        pd.Series: Group name for each club.
    """
    conditions = [
        measures["paid_transfers"] < min_paid_transfers,
        measures["balance"] > measures["band_high"],
        measures["balance"] < measures["band_low"],
    ]
    names = ["low_activity", "giving", "taking"]
    return pd.Series(np.select(conditions, names, default="two_way"), measures.index)


def count_paid_transfers(paid, club_ids):
    """Count the paid transfers each club sold plus the ones it bought.

    Args:
        paid (pd.DataFrame): Paid transfers with from_club_id and to_club_id.
        club_ids (pd.Index): All club IDs.

    Returns:
        pd.Series: Paid transfer count indexed by club ID.
    """
    sold = count_by_club(paid, "from_club_id", club_ids)
    bought = count_by_club(paid, "to_club_id", club_ids)
    return sold + bought


def build_group_measures(paid, club_ids, n_flips, band_share, seed):
    """Build the balance, random band and paid transfer count of each club.

    Args:
        paid (pd.DataFrame): Paid transfers inside one window.
        club_ids (pd.Index): All club IDs.
        n_flips (int): Number of random repetitions.
        band_share (float): Share of the random balances inside the band.
        seed (int): Seed of the random number generator.

    Returns:
        pd.DataFrame: Columns band_low, band_high, balance and paid_transfers,
            indexed by club ID.
    """
    simulated = simulate_balances(
        build_signed_fees(paid, club_ids),
        draw_directions(n_flips, len(paid), seed),
    )
    measures = compute_band(simulated, club_ids, band_share)
    measures["balance"] = compute_balance(
        sum_fees_by_club(paid, "from_club_id", club_ids),
        sum_fees_by_club(paid, "to_club_id", club_ids),
    )
    measures["paid_transfers"] = count_paid_transfers(paid, club_ids)
    return measures


def build_club_groups(
    transfers,
    club_table,
    start_date,
    end_date,
    window,
    min_paid_transfers=MIN_PAID_TRANSFERS,
    n_flips=N_FLIPS,
    band_share=BAND_SHARE,
    seed=RANDOM_SEED,
):
    """Group the clubs into giving, taking, two-way and low-activity for one window.

    Args:
        transfers (pd.DataFrame): Transfer edges with fee and is_paid columns.
        club_table (pd.DataFrame): Rows of club_id, club_name and league.
        start_date (str): First day of the window.
        end_date (str): Last day of the window.
        window (str): Label stored in the window column.
        min_paid_transfers (int): Clubs with fewer paid transfers are not grouped.
        n_flips (int): Number of random repetitions.
        band_share (float): Share of the random balances inside the band.
        seed (int): Seed of the random number generator.

    Returns:
        pd.DataFrame: One row per club with the columns in GROUP_COLUMNS.
    """
    window_transfers = filter_window(transfers, start_date, end_date)
    paid = window_transfers[window_transfers["is_paid"]]
    club_ids = pd.Index(club_table["club_id"], name="club_id")
    measures = build_group_measures(paid, club_ids, n_flips, band_share, seed)
    measures["group"] = assign_groups(measures, min_paid_transfers)
    band_columns = ["band_low", "band_high"]
    is_grouped = measures["group"] != "low_activity"
    measures[band_columns] = measures[band_columns].where(is_grouped)
    return measures.reset_index().assign(window=window)[GROUP_COLUMNS]
