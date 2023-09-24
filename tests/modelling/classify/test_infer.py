"""Test that model inference is correct on a single row"""

import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from src.modelling.classify.infer import predict_single_row


class TestPredictSingleRow(unittest.TestCase):
    """Test class to verify that a single row is predicted correctly"""

    def setUp(self):
        """Provide a single row df and dummy model for tests"""
        self.single_row_df = pd.DataFrame(
            {'apq_11': [0.1],
             'apq_3': [0.2],
             'jitter_percent': [0.2],
             'vector_similarity': [0.6]}
        )

        self.trained_model = GradientBoostingClassifier(random_state=42)
        self.trained_model.fit = MagicMock(return_value=None)

    def test_predict_single_row(self):
        """Test that a single row is predicted correctly"""
        expected_prediction = np.array([1])

        self.trained_model.predict = MagicMock(
            return_value=expected_prediction)

        prediction = predict_single_row(self.single_row_df, self.trained_model)

        np.testing.assert_array_equal(prediction, expected_prediction)
