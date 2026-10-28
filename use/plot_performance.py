import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from analyze.model_comparison import BASELINE_NAME, PREVIOUS_ONLY_NAME
from config import (
    GRID_COLOR,
    GROUP_COLORS,
    GROUP_LABELS,
    INK_COLOR,
    MUTED_COLOR,
    PERFORMANCE_FIGSIZE,
    SECONDARY_INK_COLOR,
)
from use.plot_figures import create_figure

FONT_SIZE = 14
EDGE_WIDTH = 0.5
BASELINE_FILL = "#f0efec"
TABLE_BOTTOM = 0.17
TABLE_HEIGHT = 0.66
TABLE_COLUMN_WIDTHS = [0.52, 0.16, 0.16, 0.16]
FEATURED_CLUBS = {
    281: ("Man City", (0, 16)),
    631: ("Chelsea", (-12, 0)),
    583: ("PSG", (-12, 0)),
    27: ("Bayern", (0, 16)),
    418: ("Real Madrid", (-12, 0)),
    131: ("Barcelona", (-6, 16)),
}


def create_performance_figure():
    """Create a figure with one plot in the slide style.

    Returns:
        tuple: The figure and its axes.
    """
    fig, axes = create_figure(PERFORMANCE_FIGSIZE, font_size=FONT_SIZE)
    fig.set_layout_engine("constrained")
    ax = axes[0, 0]
    ax.set_axisbelow(True)
    return fig, ax


def plot_points_by_group(groups):
    """Plot the mean points per game of each club group as bars.

    Args:
        groups (pd.DataFrame): Output of summarise_groups.

    Returns:
        plt.Figure: The figure.
    """
    fig, ax = create_performance_figure()
    labels = [f"{GROUP_LABELS[group]}\nn = {n}" for group, n in groups["club_seasons"].items()]
    bars = ax.bar(
        labels,
        groups["mean_ppg"],
        width=0.6,
        color=[GROUP_COLORS[group] for group in groups.index],
        edgecolor=SECONDARY_INK_COLOR,
        linewidth=EDGE_WIDTH,
        zorder=2,
    )
    for bar, value in zip(bars, groups["mean_ppg"]):
        ax.annotate(
            f"{value:.2f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=SECONDARY_INK_COLOR,
            fontweight="bold",
        )
    ax.set_ylabel("Points per game")
    ax.set_ylim(0, groups["mean_ppg"].max() * 1.15)
    ax.yaxis.grid(True, color=GRID_COLOR)
    ax.tick_params(axis="x", length=0)
    return fig


def label_featured_clubs(ax, clubs):
    """Label the featured clubs with a short name and a line to their point.

    Args:
        ax (plt.Axes): Axes of the scatter plot.
        clubs (pd.DataFrame): Output of summarise_clubs.
    """
    for _, row in clubs[clubs["club_id"].isin(FEATURED_CLUBS)].iterrows():
        text, (dx, dy) = FEATURED_CLUBS[row["club_id"]]
        ax.annotate(
            text,
            (row["net_spend_mio"], row["ppg"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="right" if dx < 0 else "center",
            va="center" if dy == 0 else "bottom",
            color=SECONDARY_INK_COLOR,
            zorder=5,
            arrowprops={
                "arrowstyle": "-",
                "color": MUTED_COLOR,
                "linewidth": 0.8,
                "shrinkA": 2,
                "shrinkB": 5,
            },
        )


def plot_spend_vs_points(clubs, fit):
    """Plot the mean net spend against the mean points per game, one point per club.

    Args:
        clubs (pd.DataFrame): Output of summarise_clubs.
        fit (LinregressResult): Output of fit_points_on_spend.

    Returns:
        plt.Figure: The figure.
    """
    fig, ax = create_performance_figure()
    for group in GROUP_LABELS:
        rows = clubs[clubs["group"] == group]
        ax.scatter(
            rows["net_spend_mio"],
            rows["ppg"],
            s=45,
            color=GROUP_COLORS[group],
            edgecolors=INK_COLOR,
            linewidths=EDGE_WIDTH,
            zorder=3,
        )
    x = pd.Series([clubs["net_spend_mio"].min(), clubs["net_spend_mio"].max()])
    ax.plot(x, fit.intercept + fit.slope * x, color=SECONDARY_INK_COLOR, linewidth=2, zorder=4)
    ax.axvline(0, color=MUTED_COLOR, linestyle="--", linewidth=1, zorder=1)
    ax.grid(True, color=GRID_COLOR)
    ax.set_xlabel("Mean net spend per season (million euros)")
    ax.set_ylabel("Mean points per game")
    top = clubs["ppg"].max()
    ax.set_ylim(top=top + (top - clubs["ppg"].min()) * 0.12)
    label_featured_clubs(ax, clubs)
    handles = [
        Line2D(
            [], [], marker="o", linestyle="none", markersize=9, label=label,
            markerfacecolor=GROUP_COLORS[group], markeredgecolor=INK_COLOR,
            markeredgewidth=EDGE_WIDTH,
        )
        for group, label in GROUP_LABELS.items()
    ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=len(handles),
        frameon=False,
        labelcolor=SECONDARY_INK_COLOR,
        handletextpad=0.3,
        columnspacing=1.2,
    )
    return fig


def fill_baseline_row(ax, names):
    """Fill the baseline row of the comparison table.

    Table cells with only a bottom line cannot be filled, so a rectangle sits behind the row.

    Args:
        ax (plt.Axes): Axes that hold the table.
        names (list): Model names in row order.
    """
    row_height = TABLE_HEIGHT / (len(names) + 1)
    filled = names.index(BASELINE_NAME) + 1
    bottom = TABLE_BOTTOM + TABLE_HEIGHT - (filled + 1) * row_height
    ax.add_patch(
        Rectangle((0, bottom), 1, row_height, transform=ax.transAxes,
                  facecolor=BASELINE_FILL, edgecolor="none", zorder=1)
    )


def plot_model_comparison(comparison):
    """Plot the model comparison as a plain table.

    The baseline row is filled and the previous-season-only row is bold.

    Args:
        comparison (pd.DataFrame): Output of compare_predictions.

    Returns:
        plt.Figure: The figure.
    """
    fig, ax = create_performance_figure()
    ax.axis("off")
    names = list(comparison.index)
    cell_text = [
        [name] + [f"{value:.3f}" for value in row]
        for name, row in zip(names, comparison.to_numpy())
    ]
    table = ax.table(
        cellText=cell_text,
        colLabels=["Model"] + list(comparison.columns),
        colWidths=TABLE_COLUMN_WIDTHS,
        cellLoc="center",
        bbox=[0, TABLE_BOTTOM, 1, TABLE_HEIGHT],
    )
    table.set_zorder(2)
    table.auto_set_font_size(False)
    table.set_fontsize(FONT_SIZE)
    fill_baseline_row(ax, names)
    for (row, column), cell in table.get_celld().items():
        cell.visible_edges = "B"
        cell.set_edgecolor(GRID_COLOR)
        cell.set_linewidth(0.8)
        cell.set_facecolor("none")
        cell.get_text().set_color(SECONDARY_INK_COLOR)
        if column == 0:
            cell.get_text().set_horizontalalignment("left")
            cell.PAD = 0.03
        if row == 0 or names[row - 1] == PREVIOUS_ONLY_NAME:
            cell.get_text().set_fontweight("bold")
    return fig
