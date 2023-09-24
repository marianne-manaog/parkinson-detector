"""Test file to ensure model inference and performance
evaluation are correct"""

import unittest

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from src.modelling.evaluate.inference_and_performance import infer_and_evaluate


class TestInferAndEvaluate(unittest.TestCase):
    """Class to test model inference and evaluation"""

    def setUp(self):
        """Provide dummy (Iris) data and models for tests"""
        iris = load_iris()
        x_inputs = iris.data
        y_outputs = iris.target

        random_state = 42
        x_train, x_test, y_train, y_test = train_test_split(
            x_inputs,
            y_outputs,
            test_size=0.2,
            random_state=random_state
        )

        self.gbt_classifier = GradientBoostingClassifier(
            random_state=random_state)
        self.gbt_classifier.fit(x_train, y_train)

        self.knn_classifier = KNeighborsClassifier()
        self.knn_classifier.fit(x_train, y_train)

        self.feats = pd.DataFrame(x_test, columns=iris.feature_names)
        self.targets = pd.Series(y_test)

    def test_gbt_classifier(self):
        """Test the GBT classifier"""
        preds = infer_and_evaluate(
            self.gbt_classifier,
            self.feats,
            self.targets
        )
        self.assertTrue(len(preds) == len(self.targets))

    def test_knn_classifier(self):
        """Test the KNN classifier"""
        preds = infer_and_evaluate(
            self.knn_classifier,
            self.feats,
            self.targets
        )
        self.assertTrue(len(preds) == len(self.targets))

    def test_pred_logic_with_probs(self):
        """Test the conditional logic when preds_based_on_probs is True"""
        proba_thresh = 0.5
        preds = infer_and_evaluate(
            self.gbt_classifier,
            self.feats,
            self.targets,
            preds_based_on_probs=True,
            proba_thresh=proba_thresh
        )
        self.assertTrue(len(preds) == len(self.targets))

    def test_pred_logic_without_probs(self):
        """Test the conditional logic when preds_based_on_probs is False"""
        preds = infer_and_evaluate(
            self.gbt_classifier,
            self.feats,
            self.targets,
            preds_based_on_probs=False
        )
        self.assertTrue(len(preds) == len(self.targets))
