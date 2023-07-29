# -*- coding: utf-8 -*-

"""
Usage:

    1. Make sure you have extracted features from the AI2D-RST corpus using 01_extract_features_from_corpus.py
    2. Run this script to create and plot multiple UMAP embeddings into a single figure using the following command:

        python 03_fit_multiple.py
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

# Set seaborn style
sns.set_style('ticks')

# Alternative graphs
df_files_1 = {0: {'fn': 'ai2d-rst_grouping.pkl', 'title': 'A. Expressive resources only'},
              1: {'fn': 'ai2d-rst_grouping+conn.pkl', 'title': 'B. Expressive resources and connectivity'},
              2: {'fn': 'ai2d-rst_grouping+discourse.pkl', 'title': 'C. Expressive resources and discourse semantics'}
              }

df_files_2 = {0: {'fn': 'ai2d-rst_grouping+layout.pkl', 'title': 'D. Expressive resources and layout'},
              1: {'fn': 'ai2d-rst_grouping+layout+discourse.pkl', 'title': 'E. Expressive resources, layout and discourse semantics'},
              2: {'fn': 'ai2d-rst_grouping+conn+discourse+layout.pkl', 'title': 'F. Expressive resources, layout, discourse semantics and connectivity'}
              }

# Loop over the file dictionaries
for i, file_dict in enumerate([df_files_1, df_files_2], start=1):

    # Set up figure
    fig, axs = plt.subplots(3, 1, figsize=(12, 20))

    # Loop over each DataFrame
    for k, v in file_dict.items():

        # Load DataFrame
        df = pd.read_pickle(v['fn'])

        # Initialize UMAP
        reducer = umap.UMAP(n_neighbors=60,
                            min_dist=0.1,
                            n_components=2,
                            random_state=42,
                            verbose=True,
                            metric='manhattan')

        # Fit the UMAP model
        umap_embeddings = reducer.fit_transform(df['features'].tolist())

        # Get colour map and boundaries
        cmap = plt.cm.get_cmap("tab20", 13)
        norm = mpl.colors.BoundaryNorm(np.arange(0, 13), cmap.N)

        # Initialise LabelEncoder and encode labels
        le = LabelEncoder()

        # Use the label encoder to transform class labels to numbers
        diagram_types = le.fit_transform(df['diagram_label'].tolist())

        # Plot UMAP features as points
        scatter = axs[k].scatter(umap_embeddings[:, 0],
                                 umap_embeddings[:, 1],
                                 c=diagram_types,
                                 cmap=cmap,
                                 norm=norm,
                                 s=24)

        # Set title
        axs[k].set_title(v['title'] + f" ({df['features'][0].shape[0]} dimensions)", fontsize=18)

        # Set tick params
        axs[k].tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            labelbottom=False,
            labelleft=False)

        # Create a colorbar for mapping the classes and set ticks and labels
        cbar = plt.colorbar(scatter, ticks=np.arange(0, 12) + 0.5, drawedges=True)
        cbar.ax.set_yticklabels(le.classes_, size=16)
        cbar.ax.tick_params(axis='both', which='both', size=0)

        # Set tight layout
        fig.tight_layout()

        # Plot to render everything
        plt.plot()

        # Save visualisation to disk
        fig.savefig(f"ai2d-rst_umap-{i}.pdf", bbox_inches='tight', dpi=300)
