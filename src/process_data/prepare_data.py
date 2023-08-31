"""
This file prepares the datasets to have the same input and target column names and scales.
"""

import os

import pandas as pd

from src.get_src_dir import get_src_path
from src.process_data.constants import DATA_DIR_STR, NEW_COL_NAMES
from src.process_data.utils import add_target_column, save_processed_df

ROOT_DIR_STR = str(get_src_path())


def generate_and_save_initial_processed_files() -> None:  # pragma: no cover
    """
    Generate and save the initial processed files to ensure that
    they have the same target column name ('status', as
    per the dataset of Little in 2008), and the same common
    input features identified.
    """
    little_2009_csv_path = 'Little_2009/parkinsons_updrs.csv'
    little_2009_df_processed = add_target_column(little_2009_csv_path)
    save_processed_df(little_2009_csv_path, little_2009_df_processed)

    naranjo_2016_csv_path = 'Naranjo_et_al_2016/ReplicatedAcousticFeatures-ParkinsonDatabase.csv'
    naranjo_2016_df_processed = add_target_column(naranjo_2016_csv_path)
    save_processed_df(naranjo_2016_csv_path, naranjo_2016_df_processed)

    sakar_2013_train_csv_path = 'Sakar_et_al_2013/train_data.csv'
    sakar_2013_train_df_processed = add_target_column(
        sakar_2013_train_csv_path)
    save_processed_df(sakar_2013_train_csv_path, sakar_2013_train_df_processed)

    sakar_2013_test_csv_path = 'Sakar_et_al_2013/test_data.csv'
    sakar_2013_test_df_processed = add_target_column(sakar_2013_test_csv_path)
    save_processed_df(sakar_2013_test_csv_path, sakar_2013_test_df_processed)

    sakar_2018_csv_path = 'Sakar_et_al_2018/pd_speech_features.csv'
    sakar_2018_df_processed = add_target_column(sakar_2018_csv_path)
    save_processed_df(sakar_2018_csv_path, sakar_2018_df_processed)


def process_csv(
        csv_file_path: str,
        cols_to_retain: list[str],
        new_column_names: list[str]) -> None:
    """
    Process each csv file by retaining only relevant columns and renaming them consistently,
    thus preparing them for merging them into train and test sets.

    Args:
        csv_file_path: str
            The relative path to a csv file, i.e., inside the data directory.
        cols_to_retain: list[str]
            The names of the columns to retain.
        new_column_names: list[str]
            The new names of the columns.
    """
    csv_full_path = f"{ROOT_DIR_STR}{os.sep}{DATA_DIR_STR}{os.sep}{csv_file_path}"
    data_df = pd.read_csv(csv_full_path)
    data_df = data_df[cols_to_retain]
    data_df.rename(columns=dict(
        zip(cols_to_retain, new_column_names)), inplace=True)
    data_df.to_csv(csv_full_path, index=False)


def process_speech_datasets() -> None:  # pragma: no cover
    """Process all speech datasets"""
    little_2008_csv_path = 'Little_2008/parkinsons_data.csv'
    little_2008_copy_df = pd.read_csv(
        f"{ROOT_DIR_STR}{os.sep}{DATA_DIR_STR}{os.sep}{little_2008_csv_path}")
    little_2008_copy_df_path = 'Little_2008/parkinsons_data_processed.csv'
    little_2008_copy_df.to_csv(
        f"{ROOT_DIR_STR}{os.sep}{DATA_DIR_STR}{os.sep}{little_2008_copy_df_path}",
        index=False)
    little_2008_cols_to_retain = [
        'name',
        'MDVP:Jitter(%)',
        'MDVP:Jitter(Abs)',
        'MDVP:RAP',
        'MDVP:PPQ',
        'Shimmer:APQ3',
        'Shimmer:APQ5',
        'MDVP:APQ',
        'status']
    process_csv(little_2008_copy_df_path,
                little_2008_cols_to_retain, NEW_COL_NAMES)

    little_2009_csv_path = 'Little_2009/parkinsons_updrs_processed.csv'
    little_2009_cols_to_retain = [
        'subject#',
        'Jitter(%)',
        'Jitter(Abs)',
        'Jitter:RAP',
        'Jitter:PPQ5',
        'Shimmer:APQ3',
        'Shimmer:APQ5',
        'Shimmer:APQ11',
        'status']
    process_csv(
        little_2009_csv_path,
        little_2009_cols_to_retain,
        NEW_COL_NAMES)

    naranjo_2016_csv_path = \
        'Naranjo_et_al_2016/ReplicatedAcousticFeatures-ParkinsonDatabase_processed.csv'
    naranjo_2016_cols_to_retain = [
        'ID',
        'Jitter_rel',
        'Jitter_abs',
        'Jitter_RAP',
        'Jitter_PPQ',
        'Shim_APQ3',
        'Shim_APQ5',
        'Shi_APQ11',
        'status']
    process_csv(
        naranjo_2016_csv_path,
        naranjo_2016_cols_to_retain,
        NEW_COL_NAMES)

    sakar_2013_train_csv_path = 'Sakar_et_al_2013/train_data_processed.csv'
    sakar_2013_train_cols_to_retain = [
        'Subject_id',
        'Jitter_local',
        'Jitter_local_absolute',
        'Jitter_rap',
        'Jitter_ppq5',
        'Shimmer_apq3',
        'Shimmer_apq5',
        'Shimmer_apq11',
        'status']
    process_csv(sakar_2013_train_csv_path,
                sakar_2013_train_cols_to_retain, NEW_COL_NAMES)

    sakar_2013_test_csv_path = 'Sakar_et_al_2013/test_data_processed.csv'
    sakar_2013_test_cols_to_retain = [
        'Subject_id',
        'Jitter_local',
        'Jitter_local_absolute',
        'Jitter_rap',
        'Jitter_ppq5',
        'Shimmer_apq3',
        'Shimmer_apq5',
        'Shimmer_apq11',
        'status']
    process_csv(sakar_2013_test_csv_path,
                sakar_2013_test_cols_to_retain, NEW_COL_NAMES)

    sakar_2018_csv_path = 'Sakar_et_al_2018/pd_speech_features_processed.csv'
    sakar_2018_cols_to_retain = [
        'id',
        'locPctJitter',
        'locAbsJitter',
        'rapJitter',
        'ppq5Jitter',
        'apq3Shimmer',
        'apq5Shimmer',
        'apq11Shimmer',
        'status']
    process_csv(sakar_2018_csv_path, sakar_2018_cols_to_retain, NEW_COL_NAMES)


if __name__ == '__main__':
    generate_and_save_initial_processed_files()
    process_speech_datasets()
