"""Module to visualise classification performance"""

from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from .constants import IS_4D


def visualise_class_decision_boundary(
        algo_name: str,
        label_pad: float,
        trained_model: Union[GradientBoostingClassifier, KNeighborsClassifier],
        feats: pd.DataFrame,
        targets: pd.Series,
        is_4d: bool = IS_4D
) -> None:
    """
    Visualise the classification decision boundary via a 3D scatter plot.

    Args:
        algo_name: str
            The classifier's name, e.g., KNN or GBT.
        label_pad: float
            The padding to show the z-axis label.
        trained_model: Union[GradientBoostingClassifier, KNeighborsClassifier]
            A trained model.
        feats: pd.DataFrame
            The df with features for training the KNN classifier.
        targets: pd.Series
            The column/series of target labels for supervising
            the training of the KNN classifier.
        is_4d: bool
            Whether there are four input dimensions/features
            (False by default).
    """
    # Create a mesh grid of points in the feature space
    x_min, x_max = feats.iloc[:, 0].min() - 1, feats.iloc[:, 0].max() + 1
    y_min, y_max = feats.iloc[:, 1].min() - 1, feats.iloc[:, 1].max() + 1
    z_min, z_max = feats.iloc[:, 2].min() - 1, feats.iloc[:, 2].max() + 1
    xx_mesh, yy_mesh, zz_mesh = np.meshgrid(
        np.arange(x_min, x_max, 0.1),
        np.arange(y_min, y_max, 0.1),
        np.arange(z_min, z_max, 0.1)
    )

    # Predict the class labels for the mesh grid points
    mesh_grid_points = np.c_[xx_mesh.ravel(), yy_mesh.ravel(), zz_mesh.ravel()]

    if is_4d:
        w_min, w_max = feats.iloc[:, 3].min() - 1, feats.iloc[:, 3].max() + 1
        xx_mesh, yy_mesh, zz_mesh, ww_mesh = np.meshgrid(
            np.arange(x_min, x_max, 0.1),
            np.arange(y_min, y_max, 0.1),
            np.arange(z_min, z_max, 0.1),
            np.arange(w_min, w_max, 0.1)
        )
        mesh_grid_points = np.c_[
            xx_mesh.ravel(),
            yy_mesh.ravel(),
            zz_mesh.ravel(),
            ww_mesh.ravel()]

    z_pred = trained_model.predict(mesh_grid_points)

    # Reshape z_pred to match the shape of xx, yy, zz
    z_pred = z_pred.reshape(xx_mesh.shape)

    fig = plt.figure(figsize=(14, 12))
    ax_fig = fig.add_subplot(111, projection='3d')

    train_labels_array = targets.to_numpy()
    legend_added = set()  # Keep track of added legend entries

    # Plot the decision boundary by plotting individual data points
    for label in np.unique(train_labels_array):
        indices = train_labels_array == label
        if label == 0:
            label_text = 'Healthy'
            color = 'blue'
        else:
            label_text = "Parkinson's Disease"
            color = 'red'

        if label_text not in legend_added:
            ax_fig.scatter(
                feats.iloc[indices, 0],
                feats.iloc[indices, 1],
                feats.iloc[indices, 2],
                label=f'Class: {label_text}',
                color=color,  # Set color based on class
            )
            legend_added.add(label_text)

        if is_4d:
            if label_text not in legend_added:
                ax_fig.scatter(
                    feats.iloc[indices, 0],
                    feats.iloc[indices, 1],
                    feats.iloc[indices, 2],
                    c=color,
                    # Adjust size based on the fourth dimension
                    s=40 + (feats.iloc[indices, 3] * 10),
                    label=f'Class: {label_text}'
                )
                legend_added.add(label_text)

    # Set labels and title
    ax_fig.set_xlabel(feats.columns[0])
    ax_fig.set_ylabel(feats.columns[1])
    ax_fig.set_zlabel(feats.columns[2], labelpad=label_pad)
    ax_fig.set_title(f'3D Decision Boundary of the {algo_name} Classifier')

    ax_fig.legend()

    plt.show()
