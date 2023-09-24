"""Module to assess model explainability"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import plot_tree


def plot_decision_tree(
        trained_model: GradientBoostingClassifier,
        feats: pd.DataFrame
) -> None:
    """
    Visualise model's decision tree and its
    learnt rules.

    Args:
        trained_model: GradientBoostingClassifier
            A trained model.
        feats: pd.DataFrame
            The df with features for training the KNN classifier.
    """
    tree_to_visualize = trained_model.estimators_[
        0][0]  # Access the first tree

    # Plot the decision tree
    plt.figure(figsize=(12, 8))
    # White box indicating healthy prediction at each
    # bottom leaf node, Parkinson's Disease otherwise
    plot_tree(tree_to_visualize, filled=True, feature_names=feats.columns,
              class_names=['Healthy', 'Parkinson\'s Disease'])
    plt.title('Decision Tree Visualization (GBT - First Tree)')
    plt.show()


def plot_feat_import(
        trained_model: GradientBoostingClassifier,
        feats: pd.DataFrame
) -> None:
    """
    Visualise model feature importances via a
    vertical bar chart.

    Args:
        trained_model: GradientBoostingClassifier
            A trained model.
        feats: pd.DataFrame
            The df with features for training the KNN classifier.
    """
    feature_importances = trained_model.feature_importances_

    feature_names = feats.columns

    # Sort the features by importance in descending order
    sorted_idx = np.argsort(feature_importances)[::-1]

    # Plot feature importances as a bar chart
    plt.figure(figsize=(12, 8))
    plt.bar(
        range(feats.shape[1]),
        feature_importances[sorted_idx],
        align="center"
    )
    plt.xticks(range(feats.shape[1]), [feature_names[i]
               for i in sorted_idx], rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Feature Importance")
    plt.title("GBT Feature Importances")
    plt.show()
