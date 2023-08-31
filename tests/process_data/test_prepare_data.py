"""Tests for the prepare_data module"""

import os
import unittest
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from src.get_src_dir import get_src_path
from src.process_data.constants import DATA_DIR_STR
from src.process_data.prepare_data import process_csv

ROOT_DIR_STR = str(get_src_path())


test_df = pd.DataFrame(
    {
        "col_1": [1, 2, 3],
        "col_2": [4, 5, 6]
    }
)


class TestPrepareData(unittest.TestCase):
    """Test class for the prepare_data module"""

    def setUp(self) -> None:
        """setUp method initialising and serving dummy data for tests"""
        self.first_col_name = test_df.columns[0]
        self.second_col_name = test_df.columns[1]

        self.prefix = 'new_'
        self.new_col_names = {
            self.first_col_name: f"{self.prefix}{self.first_col_name}",
            self.second_col_name: f"{self.prefix}{self.second_col_name}"
        }

        self.test_df_cols_renamed = test_df.rename(columns=self.new_col_names)

    @patch('src.process_data.prepare_data.pd.read_csv', return_value=test_df)
    @patch('src.process_data.prepare_data.pd.DataFrame.to_csv')
    def test_process_csv(self, mock_to_csv, mock_read_csv):
        """Tests to ensure df is processed with correctly renamed columns"""
        dummy_path = 'tmp/dummy_df.csv'
        full_dummy_path = f"{ROOT_DIR_STR}{os.sep}{DATA_DIR_STR}{os.sep}{dummy_path}"
        process_csv(
            dummy_path,
            test_df.columns,
            self.test_df_cols_renamed.columns)
        mock_read_csv.return_value = self.test_df_cols_renamed
        assert_frame_equal(
            self.test_df_cols_renamed,
            pd.read_csv(full_dummy_path))
        mock_to_csv.assert_called_once_with(full_dummy_path, index=False)
