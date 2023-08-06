"""
This file combines the speech datasets into train and test sets by ensuring no data leakage as follows:

- The train dataset contains only past data (from 2008 to 2013) and on the first set of subjects, since all studies
collecting each speech dataset are independent of each other
- The test dataset contains future data (from 2018) and on the second and last set of (other) subjects, i.e.,
representing 'unseen', real life-like data to the model.
"""

import os

import pandas as pd

from src.get_src_dir import get_src_path
from src.process_data.constants import DATA_DIR_STR


root_dir_str = str(get_src_path())

file_paths = {
    'little_2008_csv_path': 'Little_2008/parkinsons_data_processed.csv',
    'little_2009_csv_path': 'Little_2009/parkinsons_updrs_processed.csv',
    'naranjo_2016_csv_path': 'Naranjo_et_al_2016/ReplicatedAcousticFeatures-ParkinsonDatabase_processed.csv',
    'sakar_2013_train_csv_path': 'Sakar_et_al_2013/train_data_processed.csv',
    'sakar_2013_test_csv_path': 'Sakar_et_al_2013/test_data_processed.csv',
    'sakar_2018_csv_path': 'Sakar_et_al_2018/pd_speech_features_processed.csv'
}

dfs = {}
for key, file_path in file_paths.items():
    df_name = key.replace("_csv_path", "_df")
    dfs[df_name] = pd.read_csv(f"{root_dir_str}{os.sep}{DATA_DIR_STR}{os.sep}{file_path}")

train_data = pd.concat(
    [dfs['little_2008_df'], dfs['little_2009_df'], dfs['naranjo_2016_df'], dfs['sakar_2013_train_df'], dfs['sakar_2013_test_df']]
).dropna()

test_data = dfs['sakar_2018_df'].dropna()

train_test_data_dir_name = 'train_and_test_sets'
train_data.to_csv(
    f"{root_dir_str}{os.sep}{DATA_DIR_STR}{os.sep}{train_test_data_dir_name}{os.sep}{'train_data.csv'}",
    index=False
)
test_data.to_csv(
    f"{root_dir_str}{os.sep}{DATA_DIR_STR}{os.sep}{train_test_data_dir_name}{os.sep}{'test_data.csv'}",
    index=False
)
