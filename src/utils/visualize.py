import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from umap import UMAP


def visualize_data(x: np.ndarray, y: np.ndarray, title: str) -> go.Figure:
    """Visualize data points.

    :param x: data points
    :param y: labels
    :param title: plot title
    :return: plotly figure
    """
    df = pd.DataFrame(x, columns=["X1", "X2"])
    df["label"] = y.astype(str)

    fig = px.scatter(df, x="X1", y="X2", color="label", title=title)

    return fig


def visualize_latents(
    z: np.ndarray,
    y: np.ndarray,
    title: str,
    train_z: np.ndarray | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    return_reducer: bool = False,
) -> go.Figure:
    """Visualize latent space using UMAP.

    :param z: latent space points
    :param y: labels
    :param title: plot title
    :param train_z: training latent space points (used to fit UMAP)
    :param n_neighbors: number of neighbors
    :param min_dist: minimum distance
    :param random_state: random state
    :param return_reducer: return UMAP reducer
    :return: plotly figure (and UMAP reducer if return_reducer is True)
    """
    if train_z is None:
        train_z = z
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    reducer.fit(train_z)
    embedding = reducer.transform(z)

    visualization = visualize_data(embedding, y, title)

    if return_reducer:
        return visualization, reducer

    return visualization
