"""This Python file helps to balance samples for each class."""

import pandas as pd

from src.constants import TARGET_COL_NAME

from .constants import RANDOM_STATE


def balance_samples_per_class(
        input_df: pd.DataFrame,
        class_name: str = TARGET_COL_NAME,
        random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Balance samples per class to ensure generalisable learning/training
    in a reproducible manner.

    Args:
        input_df: pd.DataFrame
            An input df with imbalanced classes.
        class_name: str
            The class name based on which the input df needs to be balanced
            ('status' by default, indicating whether a subject has Parkinson's
            disease (if status = 1, 0 if healthy)).
        random_state: int
            The random state to draw n number of samples from the majority class
            in a reproducible manner, where n is the number of samples in the minority
            class.

    Returns:
        pd.DataFrame
            The processed df with balanced classes, i.e., with the same number
            of samples for each class.
    """

    balanced_df = input_df.copy()
    status_counts = balanced_df[class_name].value_counts()

    # Get number of rows to retain for 'class_name = 1'
    desired_count = status_counts[0]

    # Sample n rows with 'class_name = 1'
    balanced_df_train_status_1 = balanced_df[
        balanced_df[class_name] == 1
    ].sample(n=desired_count, random_state=random_state)

    balanced_df_train_status_0 = balanced_df[balanced_df[class_name] == 0]

    # Retain all balanced rows/samples
    balanced_df = pd.concat(
        [balanced_df_train_status_0, balanced_df_train_status_1])

    return balanced_df
