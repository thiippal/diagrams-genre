# -*- coding: utf-8 -*-

"""
Usage:

    1. Download the AI2D and AI2D-RST corpora as instructed in README.md
    2. Run this script to build the AI2D-RST corpus using the following command:

        python 01_extract_features_from_corpus.py
"""

# Import modules
import cv2
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from ai2d_rst import AI2D_RST
from skimage import feature
from sklearn.preprocessing import MinMaxScaler

# Set up flags for different input features or multimodal structures
incl_counts = True    # include counts for expressive resources
incl_lbp = True            # local binary patterns
incl_layout = True         # layout information
incl_connectivity = True     # connectivity information
incl_discourse = True      # discourse structure

# Set up paths to AI2D and AI2D-RST
ai2d_dir = Path("ai2d/")
ai2d_json_dir = Path("ai2d/annotations/")
ai2d_img_dir = Path("ai2d/images/")
ai2d_rst_dir = Path("ai2d/ai2d-rst")

# Set up a placeholder dictionary for the data
diagram_data = {}

# Build the grouping graphs from the AI2D-RST corpus
grouping = AI2D_RST(cat_path=ai2d_dir / 'categories_ai2d-rst.json',
                    img_path=ai2d_dir / 'images',
                    orig_json_path=ai2d_dir / 'annotations',
                    rst_json_path=ai2d_dir / 'ai2d-rst',
                    layers='grouping',
                    nx=True,
                    node_types=False)

# Extract information from the grouping graph
for i in range(0, grouping.__len__()):

    # Create placeholder dictionary for data
    grouping_data = {}

    # Retrieve the grouping graph, diagram type and filename
    graph, diagram_type, filename = grouping.__getitem__(i)

    # Use the class dictionary under the dataset to get textual labels
    grouping_data.update({'diagram_type': diagram_type[0],
                          'diagram_label': grouping.class_names[diagram_type[0]]})

    # Retrieve information on node types and features
    node_types = nx.get_node_attributes(graph, 'kind')
    node_feats = nx.get_node_attributes(graph, 'features')

    # Check if layout information should be extracted
    if incl_layout:

        # Process only arrows (1), text elements (3) and blobs (4)
        target_nodes = [k for k, v in node_types.items() if v.item() in [1, 3, 4]]
        target_feats = {k: v for k, v in node_feats.items() if k in target_nodes}

        # Extract the relative X and Y coordinates for text elements, blobs and arrows
        text_x = np.array([v[0] for k, v in target_feats.items() if graph.nodes[k]['kind'] == 3], dtype=np.float32)
        text_y = np.array([v[1] for k, v in target_feats.items() if graph.nodes[k]['kind'] == 3], dtype=np.float32)

        blob_x = np.array([v[0] for k, v in target_feats.items() if graph.nodes[k]['kind'] == 4], dtype=np.float32)
        blob_y = np.array([v[1] for k, v in target_feats.items() if graph.nodes[k]['kind'] == 4], dtype=np.float32)

        arr_x = np.array([v[0] for k, v in target_feats.items() if graph.nodes[k]['kind'] == 1], dtype=np.float32)
        arr_y = np.array([v[1] for k, v in target_feats.items() if graph.nodes[k]['kind'] == 1], dtype=np.float32)

        # Calculate a 2D histogram for position of different elements (note the order Y, X)
        text_hist = np.histogram2d(text_y, text_x, bins=[[0, 0.333, 0.666, 1.0], [0, 0.333, 0.666, 1]])
        blob_hist = np.histogram2d(blob_y, blob_x, bins=[[0, 0.333, 0.666, 1.0], [0, 0.333, 0.666, 1]])
        arr_hist = np.histogram2d(arr_y, arr_x, bins=[[0, 0.333, 0.666, 1.0], [0, 0.333, 0.666, 1]])

        # Create dictionaries for each layout 'zone' for text, blobs and arrows
        grouping_data.update({f'text_layout_{i}': np.array(x, dtype=np.float32) for i, x
                              in enumerate(text_hist[0].flatten(), start=1)})

        grouping_data.update({f'blob_layout_{i}': np.array(x, dtype=np.float32) for i, x
                              in enumerate(blob_hist[0].flatten(), start=1)})

        grouping_data.update({f'arr_layout_{i}': np.array(x, dtype=np.float32) for i, x
                              in enumerate(arr_hist[0].flatten(), start=1)})

    # Check if node counts should be included
    if incl_counts:

        # Count node types
        arr_count = [v.item() for k, v in node_types.items() if v.item() == 1]
        group_count = [v.item() for k, v in node_types.items() if v.item() == 2]
        text_count = [v.item() for k, v in node_types.items() if v.item() == 3]
        blob_count = [v.item() for k, v in node_types.items() if v.item() == 4]

        # Prepare feature vector
        grouping_data.update({'arrow_count': np.array(len(arr_count), dtype=np.float32),
                              'group_count': np.array(len(group_count), dtype=np.float32),
                              'text_count': np.array(len(text_count), dtype=np.float32),
                              'blob_count': np.array(len(blob_count), dtype=np.float32)
                              })

    # Extract local binary patterns for representing visual appearance
    img_path = str(grouping.img_path) + '/' + filename
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Check if local binary patterns should be calculated
    if incl_lbp:

        # Calculate local binary patterns
        lbp = feature.local_binary_pattern(img,
                                           24,
                                           3,
                                           method='uniform')

        # Create a normalized histogram of the local binary patterns
        (hist_lbp, _) = np.histogram(lbp,
                                     bins=range(0, 24 + 3),
                                     range=(0, 24 + 2),
                                     density=True)

        # Assign each bin in the local binary pattern histogram under its own key
        grouping_data.update({f'local_binary_pattern_{i}': np.array(x, dtype=np.float32)
                              for i, x in enumerate(hist_lbp)})

    # Update the dictionary that holds the diagram data
    diagram_data[filename] = grouping_data

# Check if connectivity information should be extracted
if incl_connectivity:

    # Build the connectivity graphs for the AI2D-RST corpus
    connectivity = AI2D_RST(cat_path=ai2d_dir / 'categories_ai2d-rst.json',
                            img_path=ai2d_dir / 'images',
                            orig_json_path=ai2d_dir / 'annotations',
                            rst_json_path=ai2d_dir / 'ai2d-rst',
                            layers='connectivity',
                            nx=True,
                            node_types=False)

    # Extract information from the connectivity graph
    for i in range(0, connectivity.__len__()):

        # Retrieve the connectivity graph, diagram type and filename
        graph, diagram_type, filename = connectivity.__getitem__(i)

        # Calculate network density (proportion of edges out of all possible edges)
        density = nx.density(graph)

        # Retrieve information on edge types
        edge_types = nx.get_edge_attributes(graph, 'kind')

        # Count directional (1), undirectional (2) and bidirectional (3) arrows
        dir_count = [v.item() for k, v in edge_types.items() if v.item() == 1]
        und_count = [v.item() for k, v in edge_types.items() if v.item() == 2]
        bid_count = [v.item() for k, v in edge_types.items() if v.item() == 3]

        # Create a dictionary containing the network data
        connectivity_data = {'directional_arrows': np.array(len(dir_count), dtype=np.float32),
                             'undirectional_arrows': np.array(len(und_count), dtype=np.float32),
                             'bidirectional_arrows': np.array(len(bid_count), dtype=np.float32)}

        # Update dictionary for the entire diagram
        if filename in diagram_data.keys():

            diagram_data[filename].update({'network_density': np.array(density, dtype=np.float32)})

            diagram_data[filename].update(connectivity_data)

        else:

            pass

# Check if discourse information should be extracted
if incl_discourse:

    # Build the discourse structure graphs for the AI2D-RST corpus
    discourse = AI2D_RST(cat_path=ai2d_dir / 'categories_ai2d-rst.json',
                         img_path=ai2d_dir / 'images',
                         orig_json_path=ai2d_dir / 'annotations',
                         rst_json_path=ai2d_dir / 'ai2d-rst',
                         layers='discourse',
                         nx=True,
                         node_types=False)

    # Extract information from the discourse structure graph
    for i in range(0, discourse.__len__()):

        # Retrieve the discourse structure graph, diagram type and filename
        graph, diagram_type, filename = discourse.__getitem__(i)

        # Retrieve information on node types
        node_types = nx.get_node_attributes(graph, 'kind')

        # Get the nodes that define RST relations
        target_nodes = [k for k, v in node_types.items() if v == 'relation']

        # Count the relation types
        relations = Counter([graph.nodes[n]['rel_name'] for n in target_nodes])

        # Convert the counts to floats for compatibility with other features
        relations = {k: np.array(v, dtype=np.float32) for k, v in relations.items()}

        # Update dictionary for the entire diagram
        if filename in diagram_data.keys():

            diagram_data[filename].update(relations)

        else:

            pass

# Create a pandas DataFrame; fill missing values with zeros
df = pd.DataFrame.from_dict(diagram_data, orient='index').fillna(np.array(0, dtype=np.float32))

for col in df.loc[:, ~df.columns.isin(['diagram_type', 'diagram_label', 'network_density'] +
                                      [col for col in df.columns
                                       if str(col).startswith('local_binary')])].columns:

    # Initialise scaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Get data and reshape for scaler
    data = np.array(df[col].tolist()).reshape(-1, 1)

    # Fit the scaler to data
    scaled_data = scaler.fit_transform(data)

    # Replace the original data with the scaled data
    df[col] = scaled_data

# Stack the individual features horizontally and store into a single column
df['features'] = df.iloc[:, 2:].apply(lambda x: np.hstack(x), axis=1)

# Save the DataFrame to disk
df.to_pickle(f'ai2d-rst_all_features.pkl')
