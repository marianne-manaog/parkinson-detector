"""Train and return a GBT classifier"""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.modelling.constants import FOLDS_NUM

from .constants import (ESTIMATORS_NUM, LEARNING_RATE, MAX_DEPTH, MAX_FEATS,
                        RANDOM_STATE, SUB_SAMPLING, TO_OPTIMISE)


def get_best_gbt_classifier(
        train_feats: pd.DataFrame,
        train_targets: pd.Series,
        to_optimise: bool = TO_OPTIMISE
) -> GradientBoostingClassifier:
    """Create and return a Gradient-Boosted Tree (GBT) classifier
    with cross-validated and optimised hyperparameters (if required).

    Args:
        train_feats: pd.DataFrame
            The df with features for training the GBT classifier.
        train_targets: pd.Series
            The column/series of target labels for supervising
            the training of the GBT classifier.
        to_optimise: bool
            Whether to optimise the model's hyperparameters.

    Returns:
        GradientBoostingClassifier
            The trained and optimised GBT classifier.
    """

    gbt = GradientBoostingClassifier(
        random_state=RANDOM_STATE,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        n_estimators=ESTIMATORS_NUM,
        max_features=MAX_FEATS,
        subsample=SUB_SAMPLING
    )

    pipeline = Pipeline(steps=[('gbt', gbt)])

    param_grid = {
        'gbt__n_estimators': [10, 20, 30, 50, 100],
        'gbt__learning_rate': [0.05, 0.15, 0.2, 0.25, 0.3],
        'gbt__max_depth': [3, 4, 5, 7, 9, 12, 15],
        'gbt__n_iter_no_change': [5, 7, 9, 11, 15],
        'gbt__subsample': [0.3, 0.5, 0.7, 0.9],
        'gbt__max_leaf_nodes': [None],
    }

    # Create a custom scorer for weighted recall
    weighted_recall_scorer = make_scorer(recall_score, average='weighted')

    if to_optimise:
        gbt = GridSearchCV(estimator=pipeline, param_grid=param_grid,
                           cv=FOLDS_NUM, scoring=weighted_recall_scorer)

    best_gbt = gbt.fit(train_feats, train_targets)
    return best_gbt
