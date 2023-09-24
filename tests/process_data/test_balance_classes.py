"""Test to ensure classes are balanced correctly"""

import unittest

import numpy as np
import pandas as pd

from src.constants import TARGET_COL_NAME
from src.process_data.balance_classes import balance_samples_per_class


class TestBalanceSamplesPerClass(unittest.TestCase):
    """Test class to ensure targets are balanced correctly"""

    def setUp(self):
        """Provide dummy data for tests"""
        self.custom_random_state = 42
        np.random.seed(self.custom_random_state)
        whole_size = 100
        self.sample_data = {
            'feature1': np.random.rand(whole_size),
            'feature2': np.random.rand(whole_size),
            TARGET_COL_NAME: np.random.choice([0, 1], size=whole_size)
        }
        self.sample_df = pd.DataFrame(self.sample_data)
        partial_size = int(whole_size / 5)
        unbalanced_data = {
            # Fewer samples in class 1
            'feature1': np.random.rand(partial_size),
            'feature2': np.random.rand(partial_size),
            TARGET_COL_NAME: np.random.choice([0, 1], size=partial_size)
        }
        self.unbalanced_df = pd.DataFrame(unbalanced_data)

    def test_balance_samples_per_class(self):
        """Unit test to ensure classes are balanced correctly"""
        balanced_df = balance_samples_per_class(self.unbalanced_df)

        class_counts = balanced_df[TARGET_COL_NAME].value_counts()
        self.assertTrue((class_counts == class_counts.iloc[0]).all())

    def test_balance_samples_per_class_random_state(self):
        """Unit test to ensure classes are balanced consistently
        when using the same (but different from the default one)
        random state"""
        balanced_df1 = balance_samples_per_class(
            self.unbalanced_df, random_state=self.custom_random_state
        )
        balanced_df2 = balance_samples_per_class(
            self.unbalanced_df, random_state=self.custom_random_state
        )

        pd.testing.assert_frame_equal(balanced_df1, balanced_df2)
