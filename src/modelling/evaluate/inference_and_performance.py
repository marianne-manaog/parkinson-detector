"""Perform inference and evaluate models' classification
performance"""

from typing import Union

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

from .constants import PREDS_WRT_PROBS


def infer_and_evaluate(
        trained_model: Union[GradientBoostingClassifier, KNeighborsClassifier],
        feats: pd.DataFrame,
        targets: pd.Series,
        preds_based_on_probs: bool = PREDS_WRT_PROBS,
        proba_thresh: float = None
) -> np.ndarray:
    """
    Serve a trained model for inference and print
    report with key classification metrics.

    Args:
        trained_model: Union[KNeighborsClassifier]
            A trained model.
        feats: pd.DataFrame
            The df with features for training the KNN classifier.
        targets: pd.Series
            The column/series of target labels for supervising
            the training of the KNN classifier.
        preds_based_on_probs: bool
            Whether getting predictions based on a custom
            probability threshold (False by default).
        proba_thresh: float
            A probability threshold based on which predictions
            are extracted (None by default).

    Returns:
        np.ndarray
            An array with the model's predictions.
    """
    preds_from_trained_model = trained_model.predict(feats)

    if preds_based_on_probs:
        preds_from_trained_model = []
        probs_from_model = trained_model.predict_proba(feats)
        for proba in probs_from_model:
            if proba[1] > proba_thresh:
                pred = 1
            else:
                pred = 0
            preds_from_trained_model.append(pred)

    print(classification_report(targets, preds_from_trained_model))

    return preds_from_trained_model
