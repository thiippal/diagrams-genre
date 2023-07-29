# -*- coding: utf-8 -*-

"""
Usage:

    1. Make sure you have extracted features from the AI2D-RST corpus using 01_extract_features_from_corpus.py
    2. Run this script to create and plot UMAP embeddings using the following command:

        python 02_fit_umap.py
"""

# Import modules
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap


# Define a function for creating thumbnails
def plot_img(path):

    img = Image.open(path)

    img.thumbnail((45, 45), Image.Resampling.LANCZOS)

    return OffsetImage(img)


# Set seaborn style
sns.set_style('ticks')

# Load the pandas DataFrame
df = pd.read_pickle('ai2d-rst_all_features.pkl')

# Create lists for UMAP parameters. Low values for 'n_neighbours' concentrate
# local structure, while larger values will focus on the global structure.
n_neighbours = [60]  # [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# The 'min_dist' defines the minimum distance between points in the low-dimensional
# representation. Small values will create local clusters, large values will prioritise
# global structure.
min_dist = [0.1 for x in range(0, len(n_neighbours))]

for n, d in zip(n_neighbours, min_dist):

    # Initialize UMAP
    reducer = umap.UMAP(n_neighbors=n,
                        min_dist=d,
                        n_components=2,
                        random_state=42,
                        verbose=True,
                        metric='manhattan')

    # Fit the UMAP model
    umap_embeddings = reducer.fit_transform(df['features'].tolist())

    # Set up matplotlib figures
    fig, ax = plt.subplots(figsize=(12, 8))
    fig_2, ax_2 = plt.subplots(figsize=(36, 24))

    # Get colour map and boundaries
    cmap = plt.get_cmap("tab20", 13)
    norm = mpl.colors.BoundaryNorm(np.arange(0, 13), cmap.N)

    # Initialise LabelEncoder and encode labels
    le = LabelEncoder()

    # Use the label encoder to transform class labels to numbers
    diagram_types = le.fit_transform(df['diagram_label'].tolist())

    # Plot UMAP features as points
    scatter = ax.scatter(umap_embeddings[:, 0],
                         umap_embeddings[:, 1],
                         c=diagram_types,
                         cmap=cmap,
                         norm=norm,
                         s=18)

    # Plot UMAP features to the second axis, otherwise thumbnails are not shown
    ax_2.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1])

    # Plot UMAP features using thumbnails
    for x, y, img in zip(umap_embeddings[:, 0],
                         umap_embeddings[:, 1],
                         df.index.tolist()):

        ab = AnnotationBbox(plot_img(f"ai2d/images/{img}"),
                            (x, y),
                            frameon=False)

        ax_2.add_artist(ab)

    # Increase label sizes
    ax_2.yaxis.set_tick_params(labelsize=24)
    ax_2.xaxis.set_tick_params(labelsize=24)

    # Plot to render everything
    plt.plot()

    # Create a colorbar for mapping the classes and set ticks and labels
    cbar = plt.colorbar(scatter, ticks=np.arange(0, 12) + 0.5)
    cbar.ax.set_yticklabels(le.classes_)
    cbar.ax.tick_params(axis='both', which='both', size=0)

    # Save visualisation to disk
    fig.savefig(f"ai2d-rst_umap_n_{n}_d_{d}_no_colours.pdf", bbox_inches='tight', dpi=300)
    fig_2.savefig(f"ai2d-rst_umap_n_{n}_d_{d}_thumbnails_no_lbp.png", bbox_inches='tight', dpi=600)
