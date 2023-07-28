"""
This file prepares the datasets to have the same input and target column names and scales.
"""

from src.process_data.utils import add_target_column, save_processed_df


def generate_and_save_processed_files() -> None:
    """
    Generate and save the processed files to ensure that they have the same target column name ('status', as per the
    dataset of Little in 2008), and the same common input features identified.
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
    generate_and_save_processed_files()
