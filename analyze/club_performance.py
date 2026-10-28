from scipy.stats import linregress

from config import GROUP_LABELS


def summarise_groups(club_seasons):
    """Compute the mean points per game and the number of club seasons of each group.

    Args:
        club_seasons (pd.DataFrame): Output of build_club_seasons.

    Returns:
        pd.DataFrame: Columns mean_ppg and club_seasons, indexed by group in the order
            of GROUP_LABELS.
    """
    summary = club_seasons.groupby("group").agg(
        mean_ppg=("ppg", "mean"),
        club_seasons=("ppg", "size"),
    )
    return summary.reindex(list(GROUP_LABELS))


def summarise_clubs(club_seasons):
    """Compute the mean net spend per season and the mean points per game of each club.

    Args:
        club_seasons (pd.DataFrame): Output of build_club_seasons.

    Returns:
        pd.DataFrame: One row per club with club_id, group, net_spend_mio and ppg.
    """
    clubs = club_seasons.groupby("club_id").agg(
        group=("group", "first"),
        net_spend_mio=("net_spend_mio", "mean"),
        ppg=("ppg", "mean"),
    )
    return clubs.reset_index()


def fit_points_on_spend(clubs):
    """Fit a line of mean points per game on mean net spend across clubs.

    Args:
        clubs (pd.DataFrame): Output of summarise_clubs.

    Returns:
        LinregressResult: Slope, intercept, rvalue (the Pearson r) and pvalue.
    """
    return linregress(clubs["net_spend_mio"], clubs["ppg"])
