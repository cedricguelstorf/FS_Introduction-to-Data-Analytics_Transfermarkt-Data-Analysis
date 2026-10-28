import numpy as np
import pandas as pd

from analyze.group_clubs import (
    build_club_groups,
    build_signed_fees,
    draw_directions,
    simulate_balances,
)
from analyze.network_analysis import filter_window
from config import (
    BAND_SHARE,
    GRID_COLOR,
    INK_COLOR,
    N_FLIPS,
    RANDOM_SEED,
    SECONDARY_INK_COLOR,
    WINDOWS,
)
from process.build_transfer_edges import build_club_table, build_transfer_edges
from use.plot_figures import create_figure, save_figure

BAND_FIGSIZE = (15.88 / 2.54, 7.63 / 2.54)
BAND_FONT_SIZE = 11
BAND_LABEL_FONT_SIZE = 10
BAR_COLOR = "#bdbcb6"
BINS = np.linspace(-1, 1, 61)
GROUP_TITLES = {"giving": "giver", "taking": "taker", "two_way": "two-way"}
EXAMPLE_GROUPS = ["giving", "two_way"]
X_TICKS = [-1, -0.5, 0, 0.5, 1]
X_TICK_LABELS = ["−1\nonly buys", "−0.5", "0", "0.5", "+1\nonly sells"]
LABEL_FLIP_BALANCE = 0.6
BAND_LABEL_DISTANCE = 0.3
LABEL_OFFSET = 0.03
HEADROOM = 1.5


def select_paid_transfers(transfers, window):
    """Select the paid transfers inside one window.

    Args:
        transfers (pd.DataFrame): Transfer edges with fee and is_paid columns.
        window (str): Key of the window in WINDOWS.

    Returns:
        pd.DataFrame: Paid transfers of the window.
    """
    window_transfers = filter_window(transfers, *WINDOWS[window])
    return window_transfers[window_transfers["is_paid"]]


def simulate_club_balances(paid, club_ids, n_flips=N_FLIPS, seed=RANDOM_SEED):
    """Simulate every club's balance under random transfer directions.

    Args:
        paid (pd.DataFrame): Paid transfers inside one window.
        club_ids (pd.Index): All club IDs, one column each.
        n_flips (int): Number of random repetitions.
        seed (int): Seed of the random number generator.

    Returns:
        np.ndarray: Balances with one row per repetition and one column per club.
    """
    signed_fees = build_signed_fees(paid, club_ids)
    directions = draw_directions(n_flips, len(paid), seed)
    return simulate_balances(signed_fees, directions)


def pick_example_club(groups, group):
    """Pick the club of one group with the most paid transfers.

    Args:
        groups (pd.DataFrame): Output of build_club_groups.
        group (str): Group name, e.g. "giving".

    Returns:
        int: Club ID.
    """
    members = groups[groups["group"] == group]
    return int(members.loc[members["paid_transfers"].idxmax(), "club_id"])


def build_club_title(club_id, groups, club_table):
    """Build the panel title of a club from its name and group.

    Args:
        club_id (int): Club ID.
        groups (pd.DataFrame): Output of build_club_groups.
        club_table (pd.DataFrame): Rows of club_id, club_name and league.

    Returns:
        str: Title such as "Club name: giver".
    """
    name = club_table.loc[club_table["club_id"] == club_id, "club_name"].iloc[0]
    group = groups.loc[groups["club_id"] == club_id, "group"].iloc[0]
    return f"{name}: {GROUP_TITLES[group]}"


def format_balance(balance):
    """Format a balance with a sign and two decimals.

    Args:
        balance (float): Balance between -1 and 1.

    Returns:
        str: Text such as "+0.53".
    """
    return f"{balance + 0.0:+.2f}".replace("-", "−")


def draw_balance_label(ax, balance):
    """Write the real balance next to its line near the top of the panel.

    Args:
        ax (plt.Axes): Axes to draw on.
        balance (float): Real balance of the club.
    """
    is_far_right = balance > LABEL_FLIP_BALANCE
    offset = -LABEL_OFFSET if is_far_right else LABEL_OFFSET
    ax.text(
        balance + offset,
        0.97,
        f"real balance\n{format_balance(balance)}",
        transform=ax.get_xaxis_transform(),
        ha="right" if is_far_right else "left",
        va="top",
        color=SECONDARY_INK_COLOR,
        fontweight="bold",
        fontsize=BAND_LABEL_FONT_SIZE,
    )


def draw_band_label(ax, band, balance, band_share):
    """Write the band label at the top of the band, away from the real balance line.

    Args:
        ax (plt.Axes): Axes to draw on.
        band (np.ndarray): Lower and upper edge of the band.
        balance (float): Real balance of the club.
        band_share (float): Share of the random balances inside the band.
    """
    is_near_left_edge = abs(balance - band[0]) < BAND_LABEL_DISTANCE
    x = band[1] if is_near_left_edge else band[0]
    ax.text(
        x + (LABEL_OFFSET if is_near_left_edge else -LABEL_OFFSET),
        0.97,
        f"{band_share:.0%} band",
        transform=ax.get_xaxis_transform(),
        ha="left" if is_near_left_edge else "right",
        va="top",
        color=SECONDARY_INK_COLOR,
        fontsize=BAND_LABEL_FONT_SIZE,
    )


def format_axes(ax, title):
    """Set title, x-axis and frame of one panel.

    Args:
        ax (plt.Axes): Axes to format.
        title (str): Panel title.
    """
    ax.set_title(title, loc="left", fontweight="bold", color=SECONDARY_INK_COLOR)
    ax.set_xlim(-1, 1)
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels(X_TICK_LABELS)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)


def draw_band_panel(ax, simulated, balance, title, band_share=BAND_SHARE):
    """Draw one club's random balances, its band and its real balance.

    Args:
        ax (plt.Axes): Axes to draw on.
        simulated (np.ndarray): The club's balance in every random repetition.
        balance (float): Real balance of the club.
        title (str): Panel title.
        band_share (float): Share of the random balances inside the band.

    Returns:
        np.ndarray: Lower and upper edge of the band.
    """
    tail = (1 - band_share) / 2 * 100
    band = np.percentile(simulated, [tail, 100 - tail])
    ax.axvspan(band[0], band[1], color=GRID_COLOR, zorder=0)
    ax.hist(simulated, bins=BINS, color=BAR_COLOR, edgecolor="white", linewidth=1, zorder=1)
    ax.axvline(balance, color=INK_COLOR, linewidth=2.5, zorder=2)
    format_axes(ax, title)
    draw_balance_label(ax, balance)
    draw_band_label(ax, band, balance, band_share)
    return band


def check_band(club_id, band, groups):
    """Raise a ValueError if the drawn band differs from the classification band.

    Args:
        club_id (int): Club ID.
        band (np.ndarray): Lower and upper edge of the drawn band.
        groups (pd.DataFrame): Output of build_club_groups.
    """
    row = groups[groups["club_id"] == club_id].iloc[0]
    if not np.allclose(band, [row["band_low"], row["band_high"]]):
        raise ValueError(f"Drawn band of club {club_id} differs from the classification band.")


def set_shared_height(axes):
    """Give all panels the same y-axis range with room for the labels.

    Args:
        axes (list): Axes to align.
    """
    top = max(ax.get_ylim()[1] for ax in axes) * HEADROOM
    for ax in axes:
        ax.set_ylim(0, top)


def plot_random_band(club_ids_to_show=None, window="full", figsize=BAND_FIGSIZE):
    """Plot the random balances and band of two clubs next to each other.

    Args:
        club_ids_to_show (list): Two club IDs. Default is the giving and the two-way
            club with the most paid transfers.
        window (str): Key of the window in WINDOWS.
        figsize (tuple): Width and height in inches.

    Returns:
        plt.Figure: The figure.
    """
    transfers = build_transfer_edges()
    club_table = build_club_table()
    club_ids = pd.Index(club_table["club_id"], name="club_id")
    paid = select_paid_transfers(transfers, window)
    groups = build_club_groups(transfers, club_table, *WINDOWS[window], window)
    simulated = simulate_club_balances(paid, club_ids)
    if club_ids_to_show is None:
        club_ids_to_show = [pick_example_club(groups, group) for group in EXAMPLE_GROUPS]
    fig, axes = create_figure(figsize, n_cols=2, font_size=BAND_FONT_SIZE)
    for ax, club_id in zip(axes[0], club_ids_to_show):
        balance = groups.loc[groups["club_id"] == club_id, "balance"].iloc[0]
        title = build_club_title(club_id, groups, club_table)
        band = draw_band_panel(ax, simulated[:, club_ids.get_loc(club_id)], balance, title)
        check_band(club_id, band, groups)
    set_shared_height(list(axes[0]))
    axes[0, 0].set_ylabel(f"random runs (of {N_FLIPS:,})")
    axes[0, 1].set_yticklabels([])
    fig.tight_layout()
    save_figure(fig, f"random_band_{window}", tight=False, transparent=True)
    return fig


if __name__ == "__main__":
    plot_random_band()
