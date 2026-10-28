import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import RANDOM_SEED, TOP_FIVE_LEAGUES

LAST_TRAIN_SEASON = 2022
N_TREES = 500
MIN_SAMPLES_LEAF = 5
BASELINE_NAME = "Baseline (previous season 1:1)"
PREVIOUS_ONLY_NAME = "OLS, previous season only"
TRANSFER_FEATURES = ["fees_paid_mio", "fees_received_mio", "previous_ppg"]


def split_by_season(club_seasons, last_train_season=LAST_TRAIN_SEASON):
    """Split the club seasons by time into earlier seasons for training and later ones for testing.

    Args:
        club_seasons (pd.DataFrame): Output of build_club_seasons.
        last_train_season (int): Last season in the training data, given by its start year.

    Returns:
        tuple: The training and the test club seasons as DataFrames.
    """
    is_train = club_seasons["season"] <= last_train_season
    return club_seasons[is_train], club_seasons[~is_train]


def build_features(club_seasons):
    """Build the transfer features, the previous points per game and the league dummies.

    The league dummies hold one 0/1 column for each league except the Premier League.

    Args:
        club_seasons (pd.DataFrame): Output of build_club_seasons.

    Returns:
        pd.DataFrame: One row per club season with the feature columns.
    """
    leagues = pd.Categorical(club_seasons["league"], categories=list(TOP_FIVE_LEAGUES))
    dummies = pd.get_dummies(leagues, prefix="league", drop_first=True, dtype=float)
    dummies.index = club_seasons.index
    return pd.concat([club_seasons[TRANSFER_FEATURES], dummies], axis=1)


def predict_test_seasons(train, test):
    """Predict the points per game of the test seasons in four ways.

    Args:
        train (pd.DataFrame): Training club seasons.
        test (pd.DataFrame): Test club seasons.

    Returns:
        dict: Model name mapped to the predicted points per game of each test club season.
    """
    train_features, test_features = build_features(train), build_features(test)
    previous_only = LinearRegression().fit(train[["previous_ppg"]], train["ppg"])
    with_transfers = LinearRegression().fit(train_features, train["ppg"])
    forest = RandomForestRegressor(
        n_estimators=N_TREES, min_samples_leaf=MIN_SAMPLES_LEAF, random_state=RANDOM_SEED
    ).fit(train_features, train["ppg"])
    return {
        BASELINE_NAME: test["previous_ppg"],
        PREVIOUS_ONLY_NAME: previous_only.predict(test[["previous_ppg"]]),
        "OLS, + transfers": with_transfers.predict(test_features),
        "Random Forest, + transfers": forest.predict(test_features),
    }


def compare_predictions(actual, predictions):
    """Compute the MAE, MSE and R² of each prediction.

    Args:
        actual (pd.Series): Actual points per game of the test club seasons.
        predictions (dict): Output of predict_test_seasons.

    Returns:
        pd.DataFrame: Columns MAE, MSE and R² rounded to 3 decimals, indexed by model.
    """
    rows = {
        name: {
            "MAE": mean_absolute_error(actual, prediction),
            "MSE": mean_squared_error(actual, prediction),
            "R²": r2_score(actual, prediction),
        }
        for name, prediction in predictions.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("Model").round(3)
