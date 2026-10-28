from analyze.build_flows import build_league_flows
from process.build_club_seasons import build_club_seasons
from process.build_transfer_edges import build_club_table, build_transfer_edges
from analyze.change_over_time import build_season_league_flows
from analyze.club_performance import fit_points_on_spend, summarise_clubs, summarise_groups
from config import GROUP_LABELS, MODEL_COMPARISON_PATH, TOP_FIVE_LEAGUES, WINDOWS
from analyze.group_clubs import build_club_groups
from analyze.model_comparison import compare_predictions, predict_test_seasons, split_by_season
from analyze.network_analysis import build_club_network_measures
from analyze.network_structure import (
    build_undirected_graph,
    build_window_graph,
    simulate_hub_model_degrees,
    simulate_random_degrees,
)
from use.plot_performance import plot_spend_vs_points, plot_model_comparison, plot_points_by_group
from use.plot_figures import (
    plot_league_fee_heatmap,
    plot_buy_sell_slide,
    plot_degree_distribution,
    plot_fees_by_league,
    plot_sales_destinations,
    save_figure,
)

SPEND_STEP_MIO = 100


def create_fees_by_league_figure():
    """Create and save the stacked fees per season chart."""
    transfers = build_transfer_edges()
    season_league_flows = build_season_league_flows(transfers)
    fig = plot_fees_by_league(season_league_flows)
    save_figure(fig, "fees_by_league", tight=False, transparent=True)


def create_sales_destinations_figure():
    """Create and save the chart of where each league's fees come from."""
    transfers = build_transfer_edges()
    league_flows = build_league_flows(transfers, *WINDOWS["full"], "full")
    fig = plot_sales_destinations(league_flows)
    save_figure(fig, "sales_destinations", tight=False, transparent=True)


def create_degree_distribution_figure():
    """Create and save the trading partners per club chart."""
    transfers = build_transfer_edges()
    club_table = build_club_table()
    measures = build_club_network_measures(transfers, club_table, *WINDOWS["full"])
    graph = build_undirected_graph(build_window_graph(transfers, club_table, *WINDOWS["full"]))
    fig = plot_degree_distribution(
        measures["partners"], simulate_random_degrees(graph), simulate_hub_model_degrees(graph)
    )
    save_figure(fig, "degree_distribution", tight=False, transparent=True)


def create_league_fee_heatmap_figure():
    """Create and save the league fee heatmap."""
    transfers = build_transfer_edges()
    league_flows = build_league_flows(transfers, *WINDOWS["full"], "full")
    fig = plot_league_fee_heatmap(league_flows)
    save_figure(fig, "league_fee_heatmap", tight=False, transparent=True)


def create_buy_sell_figures():
    """Create and save one buy-sell map for each league."""
    transfers = build_transfer_edges()
    club_table = build_club_table()
    measures = build_club_network_measures(transfers, club_table, *WINDOWS["full"])
    groups = build_club_groups(transfers, club_table, *WINDOWS["full"], "full")
    for league in TOP_FIVE_LEAGUES:
        fig = plot_buy_sell_slide(measures, groups, club_table, highlight=league)
        save_figure(fig, f"buy_sell_map_{league.lower()}", tight=False, transparent=True)


def create_points_by_group_figure():
    """Create and save the mean points per game of each club group."""
    groups = summarise_groups(build_club_seasons())
    print("Mean points per game by group:")
    for group, row in groups.iterrows():
        print(f"  {GROUP_LABELS[group]:<13} {row['mean_ppg']:.2f}  (n = {row['club_seasons']:.0f})")
    fig = plot_points_by_group(groups)
    save_figure(fig, "points_by_group", tight=False, transparent=True)


def create_spend_vs_points_figure():
    """Create and save the mean net spend against the mean points per game of each club."""
    clubs = summarise_clubs(build_club_seasons())
    fit = fit_points_on_spend(clubs)
    print(
        f"Between clubs: Pearson r = {fit.rvalue:.3f}, p = {fit.pvalue:.4f}, "
        f"slope = {fit.slope * SPEND_STEP_MIO:+.3f} points per game per {SPEND_STEP_MIO} million euros, "
        f"clubs = {len(clubs)}"
    )
    fig = plot_spend_vs_points(clubs, fit)
    save_figure(fig, "spend_vs_points", tight=False, transparent=True)


def create_model_comparison_figure():
    """Create and save the model comparison table and its CSV."""
    train, test = split_by_season(build_club_seasons())
    print(f"Training: {len(train)} club seasons, test: {len(test)} club seasons")
    comparison = compare_predictions(test["ppg"], predict_test_seasons(train, test))
    print(comparison.to_string())
    MODEL_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(MODEL_COMPARISON_PATH, float_format="%.3f")
    fig = plot_model_comparison(comparison)
    save_figure(fig, "model_comparison", tight=False, transparent=True)


if __name__ == "__main__":
    create_fees_by_league_figure()
    create_sales_destinations_figure()
    create_degree_distribution_figure()
    create_league_fee_heatmap_figure()
    create_buy_sell_figures()
    create_points_by_group_figure()
    create_spend_vs_points_figure()
    create_model_comparison_figure()
