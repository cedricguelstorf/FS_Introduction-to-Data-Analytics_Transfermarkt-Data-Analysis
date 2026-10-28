import numpy as np
import pandas as pd

from process.build_transfer_edges import build_club_table, build_transfer_edges
from config import (
    CLUB_LOCATIONS_PATH,
    COUNTRY_BOUNDS,
    KM_PER_DEGREE_LATITUDE,
    KM_PER_DEGREE_LONGITUDE_AT_EQUATOR,
    LAS_PALMAS_CLUB_ID,
    LAS_PALMAS_DISPLAY_POSITION,
    LEAGUE_COLORS,
    MAX_SPREAD_ROUNDS,
    MIN_CLUB_DISTANCE_KM,
    N_LABELLED_CLUBS,
    OUTPUT_DIR,
    REFERENCE_LATITUDE,
    TOP_FIVE_LEAGUES,
    WINDOWS,
)
from analyze.group_clubs import build_club_groups
from analyze.network_analysis import compute_balance, filter_window, sum_fees_by_club

EUROS_PER_MILLION = 1_000_000
KM_PER_DEGREE_LONGITUDE = KM_PER_DEGREE_LONGITUDE_AT_EQUATOR * np.cos(
    np.radians(REFERENCE_LATITUDE)
)
SAME_SPOT_DIRECTION = np.array([1.0, 0.0])

NODE_COLUMNS = [
    "Id",
    "Label",
    "label_short",
    "league",
    "group",
    "city",
    "latitude",
    "longitude",
    "true_latitude",
    "true_longitude",
    "shift_km",
    "fees_received_m",
    "fees_paid_m",
    "volume_m",
    "balance",
    "color_hex",
]
EDGE_COLUMNS = [
    "Source",
    "Target",
    "Type",
    "Weight",
    "transfers",
    "paying_league",
    "receiving_league",
    "is_within_league",
    "color_hex",
]

EXPECTED_NODES = 176
EXPECTED_EDGES = 2388
EXPECTED_TRANSFERS = 2993
EXPECTED_TOTAL_WEIGHT = 38509
EXPECTED_TOP_EDGES = [(583, 131, 280, 3), (631, 1237, 268, 4)]
EXPECTED_GB1_WITHIN = 8220
EXPECTED_GB1_PAID_ABROAD = 9493
EXPECTED_GB1_RECEIVED_ABROAD = 3523
EXPECTED_GROUPS = {"giving": 25, "two_way": 112, "taking": 11, "low_activity": 28}
WEIGHT_TOLERANCE = 1
MIN_DISTANCE_CHECK_KM = 44
LARGE_SHIFT_KM = 5


def to_km(latitude, longitude):
    """Convert latitude and longitude to flat positions in kilometres.

    Args:
        latitude (np.ndarray): Latitudes in degrees.
        longitude (np.ndarray): Longitudes in degrees.

    Returns:
        np.ndarray: Array with one row of x and y kilometres per position.
    """
    x = longitude * KM_PER_DEGREE_LONGITUDE
    y = latitude * KM_PER_DEGREE_LATITUDE
    return np.column_stack([x, y])


def to_degrees(points):
    """Convert flat positions in kilometres back to latitude and longitude.

    Args:
        points (np.ndarray): Array with one row of x and y kilometres per position.

    Returns:
        tuple: Latitudes and longitudes in degrees as two np.ndarray.
    """
    return points[:, 1] / KM_PER_DEGREE_LATITUDE, points[:, 0] / KM_PER_DEGREE_LONGITUDE


def find_close_pairs(points, min_distance):
    """Find the pairs of positions that are closer than the minimum distance.

    Args:
        points (np.ndarray): Array with one row of x and y kilometres per position.
        min_distance (float): Smallest allowed distance in kilometres.

    Returns:
        np.ndarray: Index pairs (first, second) with first below second.
    """
    gaps = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    return np.argwhere(np.triu(gaps < min_distance, k=1))


def push_pair_apart(points, first, second, min_distance):
    """Move two positions apart along their line, each by half the missing distance.

    Args:
        points (np.ndarray): Positions in kilometres, changed in place.
        first (int): Row of the first position.
        second (int): Row of the second position.
        min_distance (float): Smallest allowed distance in kilometres.
    """
    offset = points[second] - points[first]
    distance = np.linalg.norm(offset)
    if distance >= min_distance:
        return
    direction = offset / distance if distance > 0 else SAME_SPOT_DIRECTION
    push = (min_distance - distance) / 2 * direction
    points[first] -= push
    points[second] += push


def spread_points(points, min_distance=MIN_CLUB_DISTANCE_KM, max_rounds=MAX_SPREAD_ROUNDS):
    """Push positions apart until none is closer than the minimum distance.

    Args:
        points (np.ndarray): Array with one row of x and y kilometres per position.
        min_distance (float): Smallest allowed distance in kilometres.
        max_rounds (int): Maximum number of rounds.

    Returns:
        np.ndarray: Spread positions in the same order.
    """
    spread = points.copy()
    for _ in range(max_rounds):
        pairs = find_close_pairs(spread, min_distance)
        if len(pairs) == 0:
            break
        for first, second in pairs:
            push_pair_apart(spread, first, second, min_distance)
    return spread


def read_locations():
    """Read the stadium locations of the clubs, sorted by club ID.

    Returns:
        pd.DataFrame: Rows of club_id, city, latitude and longitude.
    """
    locations = pd.read_csv(CLUB_LOCATIONS_PATH)
    columns = ["club_id", "city", "latitude", "longitude"]
    return locations[columns].sort_values("club_id").reset_index(drop=True)


def move_las_palmas(positions):
    """Show Las Palmas at its display position so the map is not stretched.

    Args:
        positions (pd.DataFrame): Positions with club_id, latitude and longitude.

    Returns:
        pd.DataFrame: Positions with the Las Palmas row moved.
    """
    is_las_palmas = positions["club_id"] == LAS_PALMAS_CLUB_ID
    positions.loc[is_las_palmas, ["latitude", "longitude"]] = LAS_PALMAS_DISPLAY_POSITION
    return positions


def build_positions():
    """Build the true and the display position of every club.

    Returns:
        pd.DataFrame: One row per club with city, true and display coordinates
            and the spreading shift in kilometres.
    """
    locations = read_locations()
    true_points = to_km(locations["latitude"].to_numpy(), locations["longitude"].to_numpy())
    spread = spread_points(true_points)
    latitude, longitude = to_degrees(spread)
    positions = pd.DataFrame(
        {
            "club_id": locations["club_id"],
            "city": locations["city"],
            "true_latitude": locations["latitude"],
            "true_longitude": locations["longitude"],
            "latitude": latitude,
            "longitude": longitude,
            "shift_km": np.linalg.norm(spread - true_points, axis=1),
        }
    )
    return move_las_palmas(positions)


def build_volumes(paid, club_ids):
    """Sum the fees each club received and paid in million euros.

    Args:
        paid (pd.DataFrame): Paid transfers with a fee column.
        club_ids (pd.Index): All club IDs, so clubs without fees get 0.

    Returns:
        pd.DataFrame: Columns fees_received_m, fees_paid_m, volume_m and balance,
            indexed by club ID.
    """
    received = sum_fees_by_club(paid, "from_club_id", club_ids) / EUROS_PER_MILLION
    paid_out = sum_fees_by_club(paid, "to_club_id", club_ids) / EUROS_PER_MILLION
    return pd.DataFrame(
        {
            "fees_received_m": received.round(1),
            "fees_paid_m": paid_out.round(1),
            "volume_m": (received + paid_out).round(1),
            "balance": compute_balance(received, paid_out).round(3),
        }
    )


def label_largest_clubs(volume, names):
    """Keep the names of the clubs with the largest volume and blank the rest.

    Args:
        volume (pd.Series): Fee volume indexed by club ID.
        names (pd.Series): Club name indexed by club ID.

    Returns:
        pd.Series: Name for the largest clubs, empty text for the others.
    """
    largest = volume.nlargest(N_LABELLED_CLUBS).index
    return names.where(names.index.isin(largest), "")


def build_nodes(club_table, paid, groups):
    """Build one row per club with position, league, group and fee volume.

    Args:
        club_table (pd.DataFrame): Rows of club_id, club_name and league.
        paid (pd.DataFrame): Paid transfers inside the window.
        groups (pd.DataFrame): Output of build_club_groups for the same window.

    Returns:
        pd.DataFrame: One row per club with the columns in NODE_COLUMNS.
    """
    club_ids = pd.Index(club_table["club_id"], name="club_id")
    nodes = club_table.set_index("club_id")
    nodes = nodes.join(build_positions().set_index("club_id"))
    nodes = nodes.join(build_volumes(paid, club_ids))
    nodes["group"] = groups.set_index("club_id")["group"]
    nodes["Label"] = nodes["club_name"]
    nodes["label_short"] = label_largest_clubs(nodes["volume_m"], nodes["club_name"])
    nodes["color_hex"] = nodes["league"].map(LEAGUE_COLORS)
    nodes = nodes.rename_axis("Id").reset_index()
    return nodes[NODE_COLUMNS].round({"latitude": 5, "longitude": 5, "shift_km": 1})


def build_edges(paid, league_by_club):
    """Build one row per paying and receiving club pair, weighted by the fees.

    Args:
        paid (pd.DataFrame): Paid transfers with from_club_id, to_club_id and fee.
        league_by_club (pd.Series): League indexed by club ID.

    Returns:
        pd.DataFrame: Pairs with the columns in EDGE_COLUMNS, largest weight first.
    """
    pairs = paid.groupby(["to_club_id", "from_club_id"]).agg(
        fees=("fee", "sum"), transfers=("fee", "size")
    )
    pairs = pairs.reset_index()
    edges = pd.DataFrame(
        {
            "Source": pairs["to_club_id"],
            "Target": pairs["from_club_id"],
            "Type": "Directed",
            "Weight": (pairs["fees"] / EUROS_PER_MILLION).round(2),
            "transfers": pairs["transfers"],
            "paying_league": pairs["to_club_id"].map(league_by_club),
            "receiving_league": pairs["from_club_id"].map(league_by_club),
        }
    )
    edges["is_within_league"] = edges["paying_league"] == edges["receiving_league"]
    edges["color_hex"] = edges["paying_league"].map(LEAGUE_COLORS)
    return edges.sort_values("Weight", ascending=False).reset_index(drop=True)[EDGE_COLUMNS]


def expect(condition, message):
    """Print a passed check, or stop with an error if it failed.

    Args:
        condition (bool): Result of the check.
        message (str): What was checked.
    """
    if not condition:
        raise ValueError(f"Check failed: {message}")
    print(f"OK: {message}")


def check_nodes(nodes):
    """Check the node table.

    Args:
        nodes (pd.DataFrame): Output of build_nodes.
    """
    expect(len(nodes) == EXPECTED_NODES, f"{EXPECTED_NODES} nodes")
    expect(nodes["Id"].is_unique, "node IDs are unique")
    expect(nodes[["latitude", "longitude"]].notna().all().all(), "no missing position")
    expect(nodes["league"].isin(TOP_FIVE_LEAGUES).all(), "leagues are the five codes")


def check_edges(nodes, edges):
    """Check the edge table against the node table and the known totals.

    Args:
        nodes (pd.DataFrame): Output of build_nodes.
        edges (pd.DataFrame): Output of build_edges.
    """
    expect(edges["Source"].isin(nodes["Id"]).all(), "every source is a node")
    expect(edges["Target"].isin(nodes["Id"]).all(), "every target is a node")
    expect((edges["Source"] != edges["Target"]).all(), "no self-links")
    expect(len(edges) == EXPECTED_EDGES, f"{EXPECTED_EDGES} edges")
    expect(edges["transfers"].sum() == EXPECTED_TRANSFERS, f"{EXPECTED_TRANSFERS} transfers")
    total = edges["Weight"].sum()
    expect(abs(total - EXPECTED_TOTAL_WEIGHT) <= WEIGHT_TOLERANCE, f"total weight {total:.1f}")


def check_largest_edges(edges):
    """Check the largest edges.

    Args:
        edges (pd.DataFrame): Output of build_edges, largest weight first.
    """
    for row, (source, target, weight, transfers) in zip(
        edges.itertuples(), EXPECTED_TOP_EDGES
    ):
        matches = (
            (row.Source, row.Target, row.transfers) == (source, target, transfers)
            and abs(row.Weight - weight) < WEIGHT_TOLERANCE * 5
        )
        expect(matches, f"largest edge {source} to {target} with {row.Weight} million")


def check_premier_league(edges):
    """Check the money inside the Premier League and across its border.

    Args:
        edges (pd.DataFrame): Output of build_edges.
    """
    gb1_pays = edges["paying_league"] == "GB1"
    gb1_receives = edges["receiving_league"] == "GB1"
    within = edges.loc[gb1_pays & gb1_receives, "Weight"].sum()
    paid_abroad = edges.loc[gb1_pays & ~gb1_receives, "Weight"].sum()
    received_abroad = edges.loc[~gb1_pays & gb1_receives, "Weight"].sum()
    for value, expected, name in [
        (within, EXPECTED_GB1_WITHIN, "GB1 within league"),
        (paid_abroad, EXPECTED_GB1_PAID_ABROAD, "GB1 paid to other leagues"),
        (received_abroad, EXPECTED_GB1_RECEIVED_ABROAD, "GB1 received from other leagues"),
    ]:
        expect(abs(value - expected) <= WEIGHT_TOLERANCE, f"{name} {value:.1f}")


def report_groups(nodes):
    """Print the club group counts next to the counts of the reference run.

    Args:
        nodes (pd.DataFrame): Output of build_nodes.
    """
    counts = nodes["group"].value_counts().to_dict()
    print(f"Groups from the code: {counts}")
    print(f"Groups in the reference run: {EXPECTED_GROUPS}")


def min_pair_distance(nodes):
    """Find the smallest distance between two displayed clubs.

    Args:
        nodes (pd.DataFrame): Output of build_nodes.

    Returns:
        float: Smallest distance in kilometres.
    """
    points = to_km(nodes["latitude"].to_numpy(), nodes["longitude"].to_numpy())
    gaps = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    return gaps[np.triu_indices(len(points), k=1)].min()


def check_spreading(nodes):
    """Check the spreading and report how far clubs moved.

    Args:
        nodes (pd.DataFrame): Output of build_nodes.
    """
    smallest = min_pair_distance(nodes)
    expect(smallest >= MIN_DISTANCE_CHECK_KM, f"closest clubs {smallest:.1f} km apart")
    moved = int((nodes["shift_km"] > LARGE_SHIFT_KM).sum())
    largest = nodes.loc[nodes["shift_km"].idxmax()]
    print(f"Clubs moved more than {LARGE_SHIFT_KM} km: {moved}")
    print(f"Largest shift: {largest['shift_km']} km ({largest['Label']})")


def check_country_bounds(nodes):
    """Check that every true club position lies inside the box of its country.

    Args:
        nodes (pd.DataFrame): Output of build_nodes.
    """
    bounds = nodes["league"].map(COUNTRY_BOUNDS)
    inside = [
        low_lat <= lat <= high_lat and low_lon <= lon <= high_lon
        for (low_lat, high_lat, low_lon, high_lon), lat, lon in zip(
            bounds, nodes["true_latitude"], nodes["true_longitude"]
        )
    ]
    expect(all(inside), "true positions lie inside their country")


def run_checks(nodes, edges):
    """Run all checks on the node and edge tables.

    Args:
        nodes (pd.DataFrame): Output of build_nodes.
        edges (pd.DataFrame): Output of build_edges.
    """
    check_nodes(nodes)
    check_edges(nodes, edges)
    check_largest_edges(edges)
    check_premier_league(edges)
    report_groups(nodes)
    check_spreading(nodes)
    check_country_bounds(nodes)


def build_gephi_files():
    """Write the node and edge files of the full-window money network."""
    transfers = build_transfer_edges()
    club_table = build_club_table()
    window = filter_window(transfers, *WINDOWS["full"])
    paid = window[window["is_paid"]]
    groups = build_club_groups(transfers, club_table, *WINDOWS["full"], "full")
    nodes = build_nodes(club_table, paid, groups)
    edges = build_edges(paid, club_table.set_index("club_id")["league"])
    run_checks(nodes, edges)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(OUTPUT_DIR / "nodes.csv", index=False)
    edges.to_csv(OUTPUT_DIR / "edges.csv", index=False)


if __name__ == "__main__":
    build_gephi_files()
