"""Module to create a KNN classifier to
yield vector similarity"""

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from ..constants import FOLDS_NUM
from .constants import (KNN_ALGO, LEAF_SIZE, NEIGHBOURS_NUM, P_PARAM,
                        SCORING_METRIC, WEIGHTS_METHOD)


def get_best_knn_classifier(
        train_feats: pd.DataFrame,
        train_targets: pd.Series
) -> KNeighborsClassifier:
    """Create and return a K-Nearest Neighbour (KNN) classifier
    with cross-validated, optimised hyperparameter tuning.

    Args:
        train_feats: pd.DataFrame
            The df with features for training the KNN classifier.
        train_targets: pd.Series
            The column/series of target labels for supervising
            the training of the KNN classifier.

    Returns:
        KNeighborsClassifier
            The trained and optimised KNN classifier.
    """

    knn_classifier = KNeighborsClassifier(
        weights=WEIGHTS_METHOD,
        n_neighbors=NEIGHBOURS_NUM,
        p=P_PARAM,
        leaf_size=LEAF_SIZE,
        algorithm=KNN_ALGO
    )

    param_grid = {
        'weights': ['uniform', 'distance'],
        'leaf_size': [6, 8, 12, 15, 30],
        'p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
    }

    grid_search = GridSearchCV(
        estimator=knn_classifier,
        param_grid=param_grid,
        cv=FOLDS_NUM,
        scoring=SCORING_METRIC
    )

    grid_search.fit(train_feats, train_targets)

    best_knn_classifier = grid_search.best_estimator_
    return best_knn_classifier
