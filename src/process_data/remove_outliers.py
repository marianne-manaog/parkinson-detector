"""This Python file helps to identify and remove outliers."""

import numpy as np
import pandas as pd
from scipy import stats

from .constants import Z_SCORE_THRESH


def z_score_outlier_removal(
        input_df: pd.DataFrame,
        list_of_cols: list[str],
        thresh: float = Z_SCORE_THRESH
) -> pd.DataFrame:
    """Remove outliers via the z-score-based method.

    Args:
        input_df: pd.DataFrame
            An input df to filter to remove outliers.
        list_of_cols: list[str]
            A list of columns to use to process the input df.
        thresh: float
            The z-score threshold to remove outliers in the input_df.

    Returns:
        The df without outliers.
    """
    df_train_wo_outliers = input_df.copy()
    for col in list_of_cols:
        df_train_wo_outliers = df_train_wo_outliers[
            (np.abs(stats.zscore(df_train_wo_outliers[col])) < thresh)
        ]
    return df_train_wo_outliers
