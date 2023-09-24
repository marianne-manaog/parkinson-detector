"""Test file for utils sub-module of process_data module"""

import unittest
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from src.constants import TARGET_COL_NAME
from src.get_src_dir import get_src_path
from src.process_data.prepare_data import add_target_column
from src.process_data.utils import retain_selected_cols

ROOT_DIR_STR = str(get_src_path())


class TestAddTargetColumn(unittest.TestCase):
    """Test class for adding target column"""

    def setUp(self):
        """Setup method to provide required dummy data
        for tests"""
        self.test_data = {
            'class': [0, 1, 0],
            'feature_1': [1.1, 2.2, 3.3],
            'feature_2': [2.1, 3.2, 4.3]
        }
        self.expected_data = {
            TARGET_COL_NAME: [0, 1, 0],
            'feature_1': [1.1, 2.2, 3.3],
            'feature_2': [2.1, 3.2, 4.3]
        }
        self.test_df = pd.DataFrame(self.test_data)
        self.expected_df = pd.DataFrame(self.expected_data)

        self.cols_to_retain = ['class', 'feature_1']

    def test_add_target_column_existing_col(self):
        """Add target column when existing column is considered"""
        with patch('src.process_data.prepare_data.pd.read_csv', return_value=self.test_df):
            result_df = add_target_column('dummy.csv')
            assert_frame_equal(result_df, self.expected_df)

    def test_add_target_column_new_col(self):
        """Add target column when new column is considered"""
        new_col_test_data = {
            'Status': [0, 1, 0],
            'feature_1': [1.1, 2.2, 3.3],
            'feature_2': [2.1, 3.2, 4.3]
        }
        new_col_test_df = pd.DataFrame(new_col_test_data)

        with patch('src.process_data.prepare_data.pd.read_csv', return_value=new_col_test_df):
            result_df = add_target_column('dummy.csv')
            assert_frame_equal(result_df, self.expected_df)

    def test_add_target_column_no_matching_col(self):
        """Add target column when no column is matched"""
        no_matching_col_data = {
            'feature_1': [1.1, 2.2, 3.3]
        }
        no_matching_col_df = pd.DataFrame(no_matching_col_data)
        expected_no_matching_col_data = {
            'feature_1': [1.1, 2.2, 3.3],
            TARGET_COL_NAME: [1, 1, 1],
        }
        expected_no_matching_col_df = pd.DataFrame(
            expected_no_matching_col_data)

        with patch('src.process_data.prepare_data.pd.read_csv', return_value=no_matching_col_df):
            result_df = add_target_column('dummy.csv')
            assert_frame_equal(result_df, expected_no_matching_col_df)

    def test_retain_selected_cols(self):
        """Ensure only selected columns are retained"""
        result_df = retain_selected_cols(self.test_df, self.cols_to_retain)

        expected_data = {
            'class': [0, 1, 0],
            'feature_1': [1.1, 2.2, 3.3],
        }
        expected_df = pd.DataFrame(expected_data)

        pd.testing.assert_frame_equal(result_df, expected_df)
