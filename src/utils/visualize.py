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
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> go.Figure:
    """Visualize latent space using UMAP.

    :param z: latent space points
    :param y: labels
    :param title: plot title
    :return: plotly figure
    """
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    embedding = reducer.fit_transform(z)

    return visualize_data(embedding, y, title)
