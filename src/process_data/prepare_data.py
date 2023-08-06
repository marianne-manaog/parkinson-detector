"""
This file prepares the datasets to have the same input and target column names and scales.
"""

import os

import pandas as pd

from src.get_src_dir import get_src_path
from src.process_data.constants import DATA_DIR_STR
from src.process_data.utils import add_target_column, save_processed_df


root_dir_str = str(get_src_path())


def generate_and_save_initial_processed_files() -> None:
    """
    Generate and save the initial processed files to ensure that they have the same target column name ('status', as
    per the dataset of Little in 2008), and the same common input features identified.
    """
    little_2009_csv_path = 'Little_2009/parkinsons_updrs.csv'
    little_2009_df_processed = add_target_column(little_2009_csv_path)
    save_processed_df(little_2009_csv_path, little_2009_df_processed)

    naranjo_2016_csv_path = 'Naranjo_et_al_2016/ReplicatedAcousticFeatures-ParkinsonDatabase.csv'
    naranjo_2016_df_processed = add_target_column(naranjo_2016_csv_path)
    save_processed_df(naranjo_2016_csv_path, naranjo_2016_df_processed)

    sakar_2013_train_csv_path = 'Sakar_et_al_2013/train_data.csv'
    sakar_2013_train_df_processed = add_target_column(sakar_2013_train_csv_path)
    save_processed_df(sakar_2013_train_csv_path, sakar_2013_train_df_processed)

    sakar_2013_test_csv_path = 'Sakar_et_al_2013/test_data.csv'
    sakar_2013_test_df_processed = add_target_column(sakar_2013_test_csv_path)
    save_processed_df(sakar_2013_test_csv_path, sakar_2013_test_df_processed)

    sakar_2018_csv_path = 'Sakar_et_al_2018/pd_speech_features.csv'
    sakar_2018_df_processed = add_target_column(sakar_2018_csv_path)
    save_processed_df(sakar_2018_csv_path, sakar_2018_df_processed)


if __name__ == 'main':
    generate_and_save_initial_processed_files()

    new_col_names = ['subject_id', 'jitter_percent', 'jitter_abs', 'rap', 'ppq', 'apq_3', 'apq_5', 'apq_11']

    little_2008_csv_path = 'Little_2008/parkinsons_data.csv'
    little_2008_csv_full_path = f"{root_dir_str}{os.sep}{DATA_DIR_STR}{os.sep}{little_2008_csv_path}"
    little_2008_df = pd.read_csv(
        little_2008_csv_full_path
    )
    little_2008_cols_to_retain = ['name', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
     'MDVP:PPQ', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'status']
    little_2008_df = little_2008_df[little_2008_cols_to_retain]
    little_2008_df.rename(columns={new_col_names[0]: little_2008_df[little_2008_cols_to_retain[0]],
                                   new_col_names[1]: little_2008_df[little_2008_cols_to_retain[1]],
                                   new_col_names[2]: little_2008_df[little_2008_cols_to_retain[2]],
                                   new_col_names[3]: little_2008_df[little_2008_cols_to_retain[3]],
                                   new_col_names[4]: little_2008_df[little_2008_cols_to_retain[4]],
                                   new_col_names[5]: little_2008_df[little_2008_cols_to_retain[5]],
                                   new_col_names[6]: little_2008_df[little_2008_cols_to_retain[6]],
                                   new_col_names[7]: little_2008_df[little_2008_cols_to_retain[7]],
                                   }, inplace=True)
    little_2008_df.to_csv(little_2008_csv_full_path, index=False)

    little_2009_csv_path = 'Little_2009/parkinsons_updrs_processed.csv'
    little_2009_csv_full_path = f"{root_dir_str}{os.sep}{DATA_DIR_STR}{os.sep}{little_2009_csv_path}"
    little_2009_df = pd.read_csv(
        little_2009_csv_full_path
    )
    little_2009_cols_to_retain = ['subject#', 'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5',
                                  'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'status']
    little_2009_df = little_2009_df[little_2009_cols_to_retain]
    little_2009_df.rename(columns={new_col_names[0]: little_2009_df[little_2009_cols_to_retain[0]],
                                   new_col_names[1]: little_2009_df[little_2009_cols_to_retain[1]],
                                   new_col_names[2]: little_2009_df[little_2009_cols_to_retain[2]],
                                   new_col_names[3]: little_2009_df[little_2009_cols_to_retain[3]],
                                   new_col_names[4]: little_2009_df[little_2009_cols_to_retain[4]],
                                   new_col_names[5]: little_2009_df[little_2009_cols_to_retain[5]],
                                   new_col_names[6]: little_2009_df[little_2009_cols_to_retain[6]],
                                   new_col_names[7]: little_2009_df[little_2009_cols_to_retain[7]],
                                   }, inplace=True)

    naranjo_2016_csv_path = 'Naranjo_et_al_2016/ReplicatedAcousticFeatures-ParkinsonDatabase_processed.csv'
    naranjo_2016_csv_full_path = f"{root_dir_str}{os.sep}{DATA_DIR_STR}{os.sep}{naranjo_2016_csv_path}"
    naranjo_2016_df = pd.read_csv(
        naranjo_2016_csv_full_path
    )
    naranjo_2016_cols_to_retain = ['ID', 'Jitter_rel', 'Jitter_abs', 'Jitter_RAP', 'Jitter_PPQ', 'Shim_APQ3',
                                   'Shim_APQ5', 'Shi_APQ11', 'status']
    naranjo_2016_df = naranjo_2016_df[naranjo_2016_cols_to_retain]
    naranjo_2016_df.rename(columns={new_col_names[0]: naranjo_2016_df[naranjo_2016_cols_to_retain[0]],
                                   new_col_names[1]: naranjo_2016_df[naranjo_2016_cols_to_retain[1]],
                                   new_col_names[2]: naranjo_2016_df[naranjo_2016_cols_to_retain[2]],
                                   new_col_names[3]: naranjo_2016_df[naranjo_2016_cols_to_retain[3]],
                                   new_col_names[4]: naranjo_2016_df[naranjo_2016_cols_to_retain[4]],
                                   new_col_names[5]: naranjo_2016_df[naranjo_2016_cols_to_retain[5]],
                                   new_col_names[6]: naranjo_2016_df[naranjo_2016_cols_to_retain[6]],
                                   new_col_names[7]: naranjo_2016_df[naranjo_2016_cols_to_retain[7]],
                                   }, inplace=True)

    sakar_2013_csv_path = 'Sakar_et_al_2013/train_data_processed.csv'
    sakar_2013_csv_full_path = f"{root_dir_str}{os.sep}{DATA_DIR_STR}{os.sep}{sakar_2013_csv_path}"
    sakar_2013_df = pd.read_csv(
        sakar_2013_csv_full_path
    )
    sakar_2013_cols_to_retain = ['Subject id', 'Jitter (local)', 'Jitter (local, absolute)', 'Jitter (rap)',
                                 'Jitter (ppq5)', 'Shimmer (apq3)', 'Shimmer (apq5)', 'Shimmer (apq11)', 'status']
    sakar_2013_df = sakar_2013_df[sakar_2013_cols_to_retain]
    sakar_2013_df.rename(columns={new_col_names[0]: sakar_2013_df[sakar_2013_cols_to_retain[0]],
                                   new_col_names[1]: sakar_2013_df[sakar_2013_cols_to_retain[1]],
                                   new_col_names[2]: sakar_2013_df[sakar_2013_cols_to_retain[2]],
                                   new_col_names[3]: sakar_2013_df[sakar_2013_cols_to_retain[3]],
                                   new_col_names[4]: sakar_2013_df[sakar_2013_cols_to_retain[4]],
                                   new_col_names[5]: sakar_2013_df[sakar_2013_cols_to_retain[5]],
                                   new_col_names[6]: sakar_2013_df[sakar_2013_cols_to_retain[6]],
                                   new_col_names[7]: sakar_2013_df[sakar_2013_cols_to_retain[7]],
                                   }, inplace=True)

    sakar_2013_csv_path = 'Sakar_et_al_2013/test_data_processed.csv'
    sakar_2013_csv_full_path = f"{root_dir_str}{os.sep}{DATA_DIR_STR}{os.sep}{sakar_2013_csv_path}"
    sakar_2013_df = pd.read_csv(
        sakar_2013_csv_full_path
    )
    sakar_2013_cols_to_retain = ['Subject id', 'Jitter (local)', 'Jitter (local, absolute)', 'Jitter (rap)',
                                 'Jitter (ppq5)', 'Shimmer (apq3)', 'Shimmer (apq5)', 'Shimmer (apq11)', 'status']
    sakar_2013_df = sakar_2013_df[sakar_2013_cols_to_retain]
    sakar_2013_df.rename(columns={new_col_names[0]: sakar_2013_df[sakar_2013_cols_to_retain[0]],
                                   new_col_names[1]: sakar_2013_df[sakar_2013_cols_to_retain[1]],
                                   new_col_names[2]: sakar_2013_df[sakar_2013_cols_to_retain[2]],
                                   new_col_names[3]: sakar_2013_df[sakar_2013_cols_to_retain[3]],
                                   new_col_names[4]: sakar_2013_df[sakar_2013_cols_to_retain[4]],
                                   new_col_names[5]: sakar_2013_df[sakar_2013_cols_to_retain[5]],
                                   new_col_names[6]: sakar_2013_df[sakar_2013_cols_to_retain[6]],
                                   new_col_names[7]: sakar_2013_df[sakar_2013_cols_to_retain[7]],
                                   }, inplace=True)

    sakar_2018_csv_path = 'Sakar_et_al_2018/pd_speech_features.csv'
    sakar_2018_csv_full_path = f"{root_dir_str}{os.sep}{DATA_DIR_STR}{os.sep}{sakar_2018_csv_path}"
    sakar_2018_df = pd.read_csv(
        sakar_2018_csv_full_path
    )
    sakar_2018_cols_to_retain = ['id', 'locPctJitter', 'locAbsJitter', 'rapJitter', 'ppq5Jitter', 'apq3Shimmer',
                                 'apq5Shimmer', 'apq11Shimmer', 'status']
    sakar_2018_df = sakar_2018_df[sakar_2018_cols_to_retain]
    sakar_2018_df.rename(columns={new_col_names[0]: sakar_2018_df[sakar_2018_cols_to_retain[0]],
                                   new_col_names[1]: sakar_2018_df[sakar_2018_cols_to_retain[1]],
                                   new_col_names[2]: sakar_2018_df[sakar_2018_cols_to_retain[2]],
                                   new_col_names[3]: sakar_2018_df[sakar_2018_cols_to_retain[3]],
                                   new_col_names[4]: sakar_2018_df[sakar_2018_cols_to_retain[4]],
                                   new_col_names[5]: sakar_2018_df[sakar_2018_cols_to_retain[5]],
                                   new_col_names[6]: sakar_2018_df[sakar_2018_cols_to_retain[6]],
                                   new_col_names[7]: sakar_2018_df[sakar_2018_cols_to_retain[7]],
                                   }, inplace=True)
