import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, NullFormatter
from matplotlib.transforms import Bbox

from config import (
    BUY_SELL_FIGSIZE,
    DEGREE_FIGSIZE,
    GRID_COLOR,
    GROUP_LABELS,
    GROUP_MARKERS,
    FEES_FIGSIZE,
    HEATMAP_SLIDE_FIGSIZE,
    INK_COLOR,
    LEAGUE_COLORS,
    MUTED_COLOR,
    NET_BALANCE_FIGSIZE,
    OTHER_LEAGUES_COLOR,
    OUTPUT_DIR,
    SECONDARY_INK_COLOR,
    SURFACE_COLOR,
    TOP_FIVE_LEAGUES,
)

PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 13,
    "text.color": INK_COLOR,
    "axes.labelcolor": SECONDARY_INK_COLOR,
    "axes.edgecolor": MUTED_COLOR,
    "xtick.color": SECONDARY_INK_COLOR,
    "ytick.color": SECONDARY_INK_COLOR,
    "figure.facecolor": SURFACE_COLOR,
    "axes.facecolor": SURFACE_COLOR,
    "savefig.facecolor": SURFACE_COLOR,
}
FEES_MAP = LinearSegmentedColormap.from_list("fees", [GRID_COLOR, SECONDARY_INK_COLOR])
LABEL_RADII = [16, 30, 46, 64, 90, 120, 150, 180]
LABEL_ANGLES = np.deg2rad(np.arange(30, 390, 30))
MARKER_HALF_SIZE = 9
HEATMAP_BOX = [0.16, 0.14, 0.82, 0.83]


def create_figure(figsize, n_rows=1, n_cols=1, font_size=13):
    """Create a figure in the project style.

    Args:
        figsize (tuple): Width and height in inches.
        n_rows (int): Number of rows of plots.
        n_cols (int): Number of columns of plots.
        font_size (int): Base font size in points.

    Returns:
        tuple: The figure and an array of its axes.
    """
    plt.rcParams.update({**PLOT_STYLE, "font.size": font_size})
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    return fig, axes


def save_figure(fig, name, tight=True, transparent=False):
    """Save a figure as a PNG in the output folder.

    Args:
        fig (plt.Figure): Figure to save.
        name (str): File name without the extension.
        tight (bool): Crop the figure to its content, which changes its size.
        transparent (bool): Save without a background.

    Returns:
        pathlib.Path: Path of the saved file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight" if tight else None, transparent=transparent)
    return path


def place_labels(ax, points, texts, font_size=11, color=INK_COLOR, keep_inside=False,
                 obstacles=None, radii=LABEL_RADII):
    """Label points with a line to the first free spot, avoiding other labels and markers.

    Args:
        ax (plt.Axes): Axes with final limits.
        points (np.ndarray): (x, y) positions in data coordinates, one row per point.
        texts (list): One label per point.
        font_size (int): Font size of the labels.
        color (str): Colour of the labels.
        keep_inside (bool): True to keep every label inside the axes.
        obstacles (np.ndarray): (x, y) positions of other markers that labels must not cover.
        radii (list): Distances in points at which a label is tried.
    """
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    pixels = ax.transData.transform(points)
    markers = [
        Bbox.from_extents(x - MARKER_HALF_SIZE, y - MARKER_HALF_SIZE,
                          x + MARKER_HALF_SIZE, y + MARKER_HALF_SIZE)
        for x, y in pixels
    ]
    placed = []
    area = ax.get_window_extent(renderer)
    others_fixed = []
    if obstacles is not None:
        for x, y in ax.transData.transform(obstacles):
            others_fixed.append(Bbox.from_extents(x - MARKER_HALF_SIZE, y - MARKER_HALF_SIZE,
                                                  x + MARKER_HALF_SIZE, y + MARKER_HALF_SIZE))
    for index, (point, text) in enumerate(zip(points, texts)):
        others = placed + markers[:index] + markers[index + 1:] + others_fixed
        for radius in radii:
            for angle in LABEL_ANGLES:
                dx, dy = radius * np.cos(angle), radius * np.sin(angle)
                label = ax.annotate(
                    text, point, xytext=(dx, dy), textcoords="offset points",
                    fontsize=font_size, color=color,
                    ha="left" if dx >= 0 else "right", va="center", zorder=5,
                    arrowprops={"arrowstyle": "-", "color": MUTED_COLOR, "linewidth": 0.6,
                                "shrinkA": 0, "shrinkB": 4},
                )
                label.update_positions(renderer)
                box = Text.get_window_extent(label, renderer)
                is_outside = keep_inside and not (
                    area.x0 <= box.x0 and box.x1 <= area.x1
                    and area.y0 <= box.y0 and box.y1 <= area.y1
                )
                if not is_outside and not any(box.overlaps(other) for other in others):
                    placed.append(box)
                    break
                label.remove()
            else:
                continue
            break


def draw_group_markers(ax, x, y, sizes, groups, leagues, colors=None, edge=INK_COLOR):
    """Draw one marker per club: colour is the league and shape is the group.

    Args:
        ax (plt.Axes): Axes to draw on.
        x (np.ndarray): Horizontal positions.
        y (np.ndarray): Vertical positions.
        sizes (np.ndarray): Marker areas.
        groups (np.ndarray): Group name of each club.
        leagues (np.ndarray): League ID of each club.
        colors (np.ndarray): Colour of each club, or None for the league colours.
        edge (str): Edge colour of the filled markers.
    """
    if colors is None:
        colors = np.array([LEAGUE_COLORS[league] for league in leagues])
    for group, marker in GROUP_MARKERS.items():
        mask = groups == group
        if not mask.any():
            continue
        if group == "low_activity":
            ax.scatter(
                x[mask], y[mask], s=sizes[mask], marker=marker, facecolors="none",
                edgecolors=colors[mask], linewidths=1.6, zorder=3,
            )
        else:
            ax.scatter(
                x[mask], y[mask], s=sizes[mask], marker=marker, c=colors[mask],
                edgecolors=edge, linewidths=0.5, zorder=3,
            )


def build_group_handles():
    """Build the legend entries that show the shape of each group.

    Returns:
        list: One legend handle per group.
    """
    handles = []
    for group, marker in GROUP_MARKERS.items():
        face = "none" if group == "low_activity" else MUTED_COLOR
        handles.append(
            Line2D([], [], marker=marker, linestyle="none", markersize=10,
                   markerfacecolor=face, markeredgecolor=SECONDARY_INK_COLOR,
                   label=GROUP_LABELS[group])
        )
    return handles


def format_amount(value, _position):
    """Format an axis value with thousands separators.

    Args:
        value (float): Axis value.
        _position (int): Tick position, unused.

    Returns:
        str: The value as text.
    """
    return f"{value:,.0f}"


def plot_buy_sell_slide(measures, groups, club_table, highlight, figsize=BUY_SELL_FIGSIZE):
    """Plot the fees each club paid against the fees it received for a slide.

    Args:
        measures (pd.DataFrame): Output of build_club_network_measures.
        groups (pd.DataFrame): Output of build_club_groups for the same window.
        club_table (pd.DataFrame): Rows of club_id, club_name and league.
        highlight (str): League ID to colour while the other leagues are grey.
        figsize (tuple): Width and height in inches.

    Returns:
        plt.Figure: The figure.
    """
    table = measures.merge(groups[["club_id", "group"]], on="club_id")
    table = table.merge(club_table, on="club_id")
    shown = table[(table["fees_received"] > 0) & (table["fees_paid"] > 0)]
    x = shown["fees_received"].to_numpy() / 1e6
    y = shown["fees_paid"].to_numpy() / 1e6
    leagues = shown["league"].to_numpy()
    club_groups = shown["group"].to_numpy()
    fig, axes = create_figure(figsize, font_size=14)
    ax = axes[0, 0]
    ax.set_xscale("log")
    ax.set_yscale("log")
    low, high = min(x.min(), y.min()) / 1.5, max(x.max(), y.max()) * 1.5
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.plot([low, high], [low, high], color=MUTED_COLOR, linestyle="--", linewidth=1.2, zorder=1)
    sizes = np.full(len(x), 55.0)
    is_highlighted = leagues == highlight
    other = ~is_highlighted
    grey = np.full(len(x), OTHER_LEAGUES_COLOR)
    draw_group_markers(ax, x[other], y[other], sizes[other], club_groups[other],
                       leagues[other], colors=grey[other], edge=MUTED_COLOR)
    draw_group_markers(ax, x[is_highlighted], y[is_highlighted], sizes[is_highlighted],
                       club_groups[is_highlighted], leagues[is_highlighted])
    members = np.flatnonzero(is_highlighted)
    chosen = list(members[np.argsort(x[members] + y[members])[::-1][:5]])
    blocking = leagues == highlight
    blocking[chosen] = False
    ax.set_xlabel("Fees received (million euros)")
    ax.set_ylabel("Fees paid (million euros)")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(format_amount))
        axis.set_minor_formatter(NullFormatter())
    group_handles = build_group_handles() + [
        Line2D([], [], color=MUTED_COLOR, linestyle="--", label="Received = paid")
    ]
    league_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=10,
               markerfacecolor=LEAGUE_COLORS[highlight], markeredgecolor=INK_COLOR,
               markeredgewidth=0.5, label=TOP_FIVE_LEAGUES[highlight]),
        Line2D([], [], marker="o", linestyle="none", markersize=10,
               markerfacecolor=OTHER_LEAGUES_COLOR, markeredgecolor=MUTED_COLOR,
               markeredgewidth=0.5, label="Other leagues"),
    ]
    fig.legend(handles=league_handles, loc="upper left", bbox_to_anchor=(0.005, 1.0),
               ncol=5, frameon=False, labelcolor=SECONDARY_INK_COLOR, handletextpad=0.1,
               columnspacing=0.8, borderaxespad=0.2)
    fig.legend(handles=group_handles, loc="upper left", bbox_to_anchor=(0.005, 0.93), ncol=5,
               frameon=False, labelcolor=SECONDARY_INK_COLOR, handletextpad=0.1,
               columnspacing=0.8, borderaxespad=0.2)
    fig.subplots_adjust(left=0.115, right=0.99, top=0.82, bottom=0.14)
    place_labels(ax, np.column_stack([x[chosen], y[chosen]]),
                 shown["club_name"].to_numpy()[chosen], 14, color=SECONDARY_INK_COLOR,
                 keep_inside=True, obstacles=np.column_stack([x, y])[blocking])
    return fig


def build_flow_matrices(flows, window, order):
    """Pivot the long flow table into a matrix of observed flows.

    Args:
        flows (pd.DataFrame): Flow table with from, to, window and value.
        window (str): Window to keep.
        order (list): Row and column order.

    Returns:
        pd.DataFrame: Observed flows with selling labels as rows.
    """
    part = flows[flows["window"] == window]
    observed = part.pivot(index="from", columns="to", values="value")
    return observed.reindex(index=order, columns=order)


def colour_diagonal(ax, order):
    """Fill the diagonal cells of a heatmap with the colour of their league.

    Args:
        ax (plt.Axes): Heatmap axes.
        order (list): League keys in row and column order.
    """
    for position, league in enumerate(order):
        cell = Rectangle((position - 0.5, position - 0.5), 1, 1, color=LEAGUE_COLORS[league])
        ax.add_patch(cell)


def plot_league_fee_heatmap(flows, window="full", figsize=HEATMAP_SLIDE_FIGSIZE):
    """Plot the fees between the five leagues, coloured by the fees themselves.

    Args:
        flows (pd.DataFrame): League flow table.
        window (str): Window to plot.
        figsize (tuple): Width and height in inches.

    Returns:
        plt.Figure: The figure.
    """
    order = list(TOP_FIVE_LEAGUES)
    observed = build_flow_matrices(flows, window, order)
    fig, axes = create_figure(figsize, font_size=14)
    ax = axes[0, 0]
    ax.set_position(HEATMAP_BOX)
    shade = observed / observed.to_numpy().max()
    ax.imshow(shade, cmap=FEES_MAP, vmin=0, vmax=1, aspect="auto")
    colour_diagonal(ax, order)
    strong = (shade > 0.55) | np.eye(len(order), dtype=bool)
    for row in range(len(order)):
        for column in range(len(order)):
            ax.text(column, row, f"{observed.iloc[row, column] / 1e6:,.0f}", ha="center",
                    va="center", color="white" if strong.iloc[row, column] else SECONDARY_INK_COLOR)
    ax.set_xticks(range(len(order)), order)
    ax.set_yticks(range(len(order)), order)
    ax.set_xticks(np.arange(-0.5, len(order)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order)), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="both", length=0)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.set_xlabel("Buying league")
    ax.set_ylabel("Selling league")
    return fig


def plot_degree_distribution(partners, random_degrees, hub_degrees, figsize=DEGREE_FIGSIZE):
    """Plot the number of trading partners per club against random and hub-model networks.

    Args:
        partners (pd.Series): Trading partners per club.
        random_degrees (np.ndarray): Erdos-Renyi degrees with one row per network.
        hub_degrees (np.ndarray): Barabasi-Albert degrees with one row per network.
        figsize (tuple): Width and height in inches.

    Returns:
        plt.Figure: The figure.
    """
    edges = np.arange(0, 128, 4)
    centres = (edges[:-1] + edges[1:]) / 2
    real_counts, _ = np.histogram(partners, bins=edges)
    random_counts = np.mean([np.histogram(row, bins=edges)[0] for row in random_degrees], axis=0)
    hub_counts = np.mean([np.histogram(row, bins=edges)[0] for row in hub_degrees], axis=0)
    fig, axes = create_figure(figsize, font_size=14)
    ax = axes[0, 0]
    ax.bar(centres, real_counts, width=3.4, color=MUTED_COLOR, label="Real network", zorder=2)
    ax.plot(centres, random_counts, color=SECONDARY_INK_COLOR, linewidth=2.5,
            label="Erd\u0151s\u2013R\u00e9nyi", zorder=3)
    ax.plot(centres, hub_counts, color=SECONDARY_INK_COLOR, linewidth=2.5,
            linestyle="--", label="Barab\u00e1si\u2013Albert", zorder=3)
    ax.set_xlabel("Trading partners per club")
    ax.set_ylabel("Number of clubs")
    ax.set_xticks(np.arange(0, 125, 20))
    ax.yaxis.grid(True, color=GRID_COLOR)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, labelcolor=SECONDARY_INK_COLOR)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.98, bottom=0.13)
    return fig


def build_fees_paid_by_league(season_league_flows):
    """Pivot the fees each league paid into one row per season.

    Args:
        season_league_flows (pd.DataFrame): Output of build_season_league_flows.

    Returns:
        pd.DataFrame: Fees in euros with seasons as rows and league IDs as columns.
    """
    paid = season_league_flows.groupby(["window", "to"])["value"].sum().unstack()
    return paid[list(TOP_FIVE_LEAGUES)]


def plot_fees_by_league(season_league_flows, covid_season="2020_21", figsize=FEES_FIGSIZE):
    """Plot the fees of each season as stacked bars, split by the league that paid them.

    Args:
        season_league_flows (pd.DataFrame): Output of build_season_league_flows.
        covid_season (str): Season marked as the COVID-19 season.
        figsize (tuple): Width and height in inches.

    Returns:
        plt.Figure: The figure.
    """
    paid = build_fees_paid_by_league(season_league_flows) / 1e9
    seasons = paid.index.tolist()
    positions = np.arange(len(seasons))
    fig, axes = create_figure(figsize, font_size=14)
    ax = axes[0, 0]
    bottom = np.zeros(len(seasons))
    for league, name in TOP_FIVE_LEAGUES.items():
        ax.bar(positions, paid[league], bottom=bottom, width=0.7, color=LEAGUE_COLORS[league],
               label=name, zorder=2)
        bottom += paid[league].to_numpy()
    for position, total in zip(positions, bottom):
        ax.text(position, total + 0.06, f"{total:.1f}", ha="center", va="bottom",
                color=SECONDARY_INK_COLOR, fontweight="bold")
    covid_position = seasons.index(covid_season)
    ax.axvspan(covid_position - 0.5, covid_position + 0.5, color=GRID_COLOR, alpha=0.6, zorder=0)
    ax.text(covid_position, bottom.max() * 1.12, "COVID-19", ha="center", va="bottom",
            color=SECONDARY_INK_COLOR)
    ax.set_xticks(positions, [f"{season[2:4]}/{season[5:]}" for season in seasons])
    ax.set_ylim(0, bottom.max() * 1.2)
    ax.set_ylabel("Fees paid (billion euros)")
    ax.yaxis.grid(True, color=GRID_COLOR)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 1.0), ncol=5,
               frameon=False, labelcolor=SECONDARY_INK_COLOR, handlelength=1,
               handletextpad=0.4, columnspacing=0.8, borderaxespad=0.2)
    fig.subplots_adjust(left=0.1, right=0.995, top=0.87, bottom=0.12)
    return fig


def build_sales_destinations(league_flows):
    """Compute the share of each league's fees received that comes from each buying league.

    Args:
        league_flows (pd.DataFrame): Output of build_league_flows for the full window.

    Returns:
        pd.DataFrame: Shares in percent with selling leagues as rows and buying leagues as columns.
    """
    sales = league_flows.pivot_table(index="from", columns="to", values="value", aggfunc="sum")
    sales = sales.loc[list(TOP_FIVE_LEAGUES), list(TOP_FIVE_LEAGUES)]
    return sales.div(sales.sum(axis=1), axis=0) * 100


def plot_sales_destinations(league_flows, figsize=NET_BALANCE_FIGSIZE):
    """Plot where the fees of each selling league come from as stacked bars of 100 percent.

    Args:
        league_flows (pd.DataFrame): Output of build_league_flows for the full window.
        figsize (tuple): Width and height in inches.

    Returns:
        plt.Figure: The figure.
    """
    shares = build_sales_destinations(league_flows)
    home = pd.Series({league: shares.loc[league, league] for league in shares.index})
    abroad = (100 - home).sort_values(ascending=False)
    positions = np.arange(len(abroad))
    fig, axes = create_figure(figsize, font_size=14)
    ax = axes[0, 0]
    for position, seller in zip(positions, abroad.index):
        bottom = 0
        buyers = [seller] + [league for league in TOP_FIVE_LEAGUES if league != seller]
        for buyer in buyers:
            ax.bar(position, shares.loc[seller, buyer], bottom=bottom, width=0.65,
                   color=OTHER_LEAGUES_COLOR if buyer == seller else LEAGUE_COLORS[buyer], zorder=2)
            bottom += shares.loc[seller, buyer]
    for position, value in zip(positions, abroad):
        ax.text(position, 102, f"{value:.0f}%", ha="center", va="bottom",
                color=SECONDARY_INK_COLOR, fontweight="bold")
    names = [textwrap.fill(TOP_FIVE_LEAGUES[league], 11) for league in abroad.index]
    ax.set_xticks(positions, names)
    ax.set_xlim(-0.5, len(abroad) - 0.5)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Share of fees received (%)")
    ax.yaxis.grid(True, color=GRID_COLOR)
    ax.set_axisbelow(True)
    handles = [Patch(facecolor=LEAGUE_COLORS[league]) for league in TOP_FIVE_LEAGUES]
    handles.append(Patch(facecolor=OTHER_LEAGUES_COLOR))
    fig.legend(handles, list(TOP_FIVE_LEAGUES.values()) + ["Own league"], title="Fees paid by", loc="upper left",
               bbox_to_anchor=(0.005, 1.0), ncol=3, frameon=False, labelcolor=SECONDARY_INK_COLOR,
               handlelength=1, handletextpad=0.5, columnspacing=1.0, borderaxespad=0.2,
               alignment="left").get_title().set_color(SECONDARY_INK_COLOR)
    fig.subplots_adjust(left=0.14, right=0.995, top=0.76, bottom=0.13)
    return fig
