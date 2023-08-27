"""
This Python file analyses the five speech datasets in this project.
"""

import logging
import os

import pandas as pd

from src.constants import TARGET_COL_NAME
from src.get_src_dir import get_src_path
from src.process_data.constants import DATA_DIR_STR

logging.basicConfig(level=logging.INFO)

ROOT_DIR_STR = str(get_src_path())


def analyse_dataset(csv_path: str) -> None:
    """
    Analyse a speech dataset and log its key characteristics,
    such as column names, descriptive statistics and
    which columns have missing values.

    Args:
        csv_path: str
                The path to a csv dataset.
    """

    data_df = pd.read_csv(csv_path)

    logging.info(f"The first two rows of the df are: {data_df.head(2)}.")

    cols = data_df.columns
    logging.info(f"The columns of the df are: {cols}.")

    for col in cols:
        logging.info(
            f"The descriptive statistics of the df are: {data_df[col].describe()}.")

    list_cols_nan = data_df.columns[data_df.isnull().any()].tolist()
    if len(list_cols_nan) > 0:
        logging.info(f"The missing values of the df are: {list_cols_nan}.")
    else:
        logging.info("There are no columns with missing values.")

    # 'status' column (0 = healthy, 1 = PD)
    logging.info(
        f"The number of healthy vs PD subjects of the "
        f"df is: {data_df[TARGET_COL_NAME].value_counts()}.")


if __name__ == '__main__':
    list_of_paths_of_files_to_analyse = [
        'Little_2008/parkinsons_data.csv',
        'Little_2009/parkinsons_updrs_processed.csv',
        'Naranjo_et_al_2016/ReplicatedAcousticFeatures-ParkinsonDatabase_processed.csv',
        'Sakar_et_al_2013/train_data_processed.csv',
        'Sakar_et_al_2013/test_data_processed.csv',
        'Sakar_et_al_2018/pd_speech_features_processed.csv']

    for file_path in list_of_paths_of_files_to_analyse:
        analyse_dataset(
            f"{ROOT_DIR_STR}{os.sep}{DATA_DIR_STR}{os.sep}{file_path}")
