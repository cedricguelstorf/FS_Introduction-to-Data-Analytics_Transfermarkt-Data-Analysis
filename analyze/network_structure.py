import networkx as nx
import numpy as np

from config import N_REFERENCE_NETWORKS, RANDOM_SEED
from analyze.network_analysis import build_club_graph, filter_window


def build_window_graph(transfers, club_table, start_date, end_date):
    """Build the directed club graph of one window.

    The weight of an edge is its number of transfers.

    Args:
        transfers (pd.DataFrame): Transfer edges with club ID columns.
        club_table (pd.DataFrame): Rows of club_id, club_name and league.
        start_date (str): First day of the window.
        end_date (str): Last day of the window.

    Returns:
        nx.DiGraph: One node per club, one edge per selling and buying club pair.
    """
    window_transfers = filter_window(transfers, start_date, end_date)
    clubs = club_table.set_index("club_id")
    return build_club_graph(window_transfers, clubs["league"], clubs["club_name"])


def build_undirected_graph(graph):
    """Merge the two directions of each club pair into one edge.

    Args:
        graph (nx.DiGraph): Output of build_window_graph.

    Returns:
        nx.Graph: One edge per club pair, weighted by the transfers in both
            directions.
    """
    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes(data=True))
    for source, target, data in graph.edges(data=True):
        earlier = undirected.get_edge_data(source, target, {"weight": 0})["weight"]
        undirected.add_edge(source, target, weight=data["weight"] + earlier)
    return undirected


def build_random_undirected_graph(n_nodes, n_edges, seed):
    """Build an Erdos-Renyi graph.

    Args:
        n_nodes (int): Number of clubs.
        n_edges (int): Number of edges.
        seed (int): Seed of the random number generator.

    Returns:
        nx.Graph: Random graph with the given number of nodes and edges.
    """
    return nx.gnm_random_graph(n_nodes, n_edges, seed=seed)


def simulate_random_degrees(real_graph, n_networks=N_REFERENCE_NETWORKS, seed=RANDOM_SEED):
    """Collect the degrees of Erdos-Renyi networks with as many links as the real one.

    Args:
        real_graph (nx.Graph): Output of build_undirected_graph.
        n_networks (int): Number of random networks.
        seed (int): Seed of the first network, the next ones count up from it.

    Returns:
        np.ndarray: Degrees with one row per random network and one column per club.
    """
    degrees = []
    for draw in range(n_networks):
        graph = build_random_undirected_graph(
            real_graph.number_of_nodes(), real_graph.number_of_edges(), seed + draw
        )
        degrees.append([degree for _, degree in graph.degree()])
    return np.array(degrees)


def simulate_hub_model_degrees(real_graph, n_networks=N_REFERENCE_NETWORKS, seed=RANDOM_SEED):
    """Collect the degrees of Barabasi-Albert networks with about as many links as the real one.

    Args:
        real_graph (nx.Graph): Output of build_undirected_graph.
        n_networks (int): Number of Barabasi-Albert networks.
        seed (int): Seed of the first network, the next ones count up from it.

    Returns:
        np.ndarray: Degrees with one row per network and one column per club.
    """
    n_nodes = real_graph.number_of_nodes()
    links_per_node = min(
        range(1, n_nodes),
        key=lambda m: abs(m * (n_nodes - m) - real_graph.number_of_edges()),
    )
    degrees = []
    for draw in range(n_networks):
        graph = nx.barabasi_albert_graph(n_nodes, links_per_node, seed=seed + draw)
        degrees.append([degree for _, degree in graph.degree()])
    return np.array(degrees)
