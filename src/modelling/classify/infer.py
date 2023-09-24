"""Perform model inference"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


def predict_single_row(
        single_row_df: pd.DataFrame,
        trained_model: GradientBoostingClassifier
) -> np.ndarray:
    """
    Generate a prediction based on a single row df.

    Args:
        single_row_df: pd.DataFrame
            A single row df, such as:
                pd.DataFrame(
                    {'apq_11': [0.1], 'apq_3': [0.2], 'jitter_percent': [0.2],
                    'vector_similarity': [0.6], }
                )
        trained_model: GradientBoostingClassifier
            A trained GBT model.

    Returns:
        np.ndarray
            A prediction as a numpy array.
    """
    prediction = trained_model.predict(single_row_df)
    return prediction
