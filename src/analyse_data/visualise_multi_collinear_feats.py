"""This Python file helps in visualising multi-collinear features
via an appropriate correlation analysis"""

import matplotlib.pyplot as plt
import pandas as pd

from .constants import NON_PARAM_CORR_METHOD


def plot_multi_collinearity_heatmap(
        input_df: pd.DataFrame,
        corr_method: str = NON_PARAM_CORR_METHOD
) -> None:
    """Visualise multi-collinear features via a heatmap
    using a correlation analysis.

    Args:
        input_df: pd.DataFrame
            An input dataframe with various numerical features to
            identify multi-collinear ones.
        corr_method: str
            The chosen correlation method (by default, using the
            non-parametric Kendall's tau method).
    """
    # Calculate the correlation matrix on the numeric columns
    correl_matrix = input_df.corr(method=corr_method)

    # Create a heatmap with correlation strengths as annotations
    fig, ax_fig = plt.subplots(figsize=(8, 6))
    cax = ax_fig.matshow(correl_matrix, cmap='coolwarm')
    _ = fig.colorbar(cax)

    # Add annotations
    for i in range(correl_matrix.shape[0]):
        for j in range(correl_matrix.shape[1]):
            ax_fig.text(
                j,
                i,
                f'{correl_matrix.iloc[i, j]:.2f}',
                ha='center',
                va='center',
                color='black',
                fontsize=10
            )

    plt.xticks(
        range(len(correl_matrix.columns)),
        correl_matrix.columns,
        rotation=45
    )
    plt.yticks(
        range(len(correl_matrix.columns)),
        correl_matrix.columns
    )

    plt.show()
