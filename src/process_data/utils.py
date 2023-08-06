"""
Utility-based functions to aid data preparation.
"""

import os

import pandas as pd

from src.get_src_dir import get_src_path
from src.process_data.constants import DATA_DIR_STR, PROCESSED_SUFFIX, CSV_FMT_STR
from src.constants import TARGET_COL_NAME

root_dir_str = str(get_src_path())


def add_target_column(initial_file_path: str, target_col_name: str = TARGET_COL_NAME) -> pd.DataFrame:
    """
    Add the target column to an initial df given its file path.

    Args:
        initial_file_path: str
                The path to the initial csv file (with the extension).
        target_col_name: str
                The name of the target column ('status' by default).

    Returns:
        The updated df with the target column.
    """

    initial_df = pd.read_csv(
        f"{root_dir_str}{os.sep}{DATA_DIR_STR}{os.sep}{initial_file_path}"
    )

    updated_df = initial_df.copy()

    expected_cols_tuple = ('class', 'Status')

    df_cols = initial_df.columns
    if any(s in df_cols for s in expected_cols_tuple):
        if expected_cols_tuple[0] in df_cols:
            # Rename col for consistency wrt other 'status' cols in the other dfs
            updated_df.rename(columns={expected_cols_tuple[0]: target_col_name}, inplace=True)
        elif expected_cols_tuple[1] in df_cols:
            # Rename col for consistency wrt other 'status' cols in the other dfs
            updated_df.rename(columns={expected_cols_tuple[1]: target_col_name}, inplace=True)
    else:
        # Creating a new column called 'status' with 1s because all rows pertain to PD subjects (for the data of Max
        # Little in 2009)
        updated_df[target_col_name] = [1] * len(initial_df)

    return updated_df


def save_processed_df(
        initial_file_path: str, processed_df: pd.DataFrame, processed_suffix: str = PROCESSED_SUFFIX
) -> None:
    """
    Save the processed df to a csv file and add a suffix to differentiate it from the initial file.

    Args:
        initial_file_path: str
                The path to the initial csv file (with the extension).
        processed_df: pd.DataFrame
                The processed df to save into a csv file.
        processed_suffix: str
                The processed file name's suffix ('_processed' by default).
    """
    processed_df.to_csv(
        f"{root_dir_str}{os.sep}{DATA_DIR_STR}{os.sep}{initial_file_path.split('.')[0]}{processed_suffix}{CSV_FMT_STR}",
        index=False
    )
