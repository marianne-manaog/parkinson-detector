"""Test file to ensure outliers are removed correctly"""

import unittest

import pandas as pd

from src.process_data.remove_outliers import z_score_outlier_removal


class TestZScoreOutlierRemoval(unittest.TestCase):
    """Test class to ensure outliers are removed correctly"""

    def setUp(self) -> None:
        """Create dummy data for tests"""
        data = {
            'A': [1, 2, 3, 100, 5, 6, 7, 8],
            'B': [10, 20, 30, 40, 50, 60, 70, 80]
        }
        self.input_df = pd.DataFrame(data)

        # Define the columns to process and the z-score threshold
        self.list_of_cols = ['A', 'B']
        self.thresh = 2.0

    def test_outlier_removal(self):
        """Unit test to ensure outliers are removed correctly"""

        result_df = z_score_outlier_removal(
            self.input_df,
            self.list_of_cols,
            self.thresh
        ).reset_index(drop=True)

        expected_data = {
            'A': [1, 2, 3, 5, 6, 7, 8],
            'B': [10, 20, 30, 50, 60, 70, 80]
        }
        expected_df = pd.DataFrame(expected_data).reset_index(drop=True)

        pd.testing.assert_frame_equal(result_df, expected_df)
