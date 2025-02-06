import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

import torch
import networkx as nx
from torch_geometric.utils import to_networkx
from scipy.stats import multivariate_normal


def plot_cluster(adata_,size=15,
                color_list = [
                    '#f94144',
                    '#f3722c',
                    '#f8961e',
                    '#f9c74f',
                    '#90be6d',
                    '#43aa8b',
                    '#577590',  
                    '#226CE0',
                    '#77479F'

                     ],
                     legd = False,
                     return_=False):
    fig, ax = plt.subplots(1,1,figsize=(3,6),dpi=800)
    for j,i in enumerate(adata_.obs['cluster'].unique()):
        plt.scatter(adata_.obsm['spatial'][adata_.obs['cluster']==i,0],
                    adata_.obsm['spatial'][adata_.obs['cluster']==i,1],
                    s=size,
                    color = color_list[j]
                   )
    if legd:
        plt.legend(adata_.obs['cluster'].unique(),bbox_to_anchor=(1.01,0.8))
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])  
    
def downsample_and_plot_spatial_data(adata, color_list=None, n_blocks_x=5, n_blocks_y=5,size=15, figsize=(3,6),invert_yaxis=True):
    spatial_data = adata.obsm['spatial']
    min_coord = spatial_data.min(axis=0)
    max_coord = spatial_data.max(axis=0)
    norm_spatial = (spatial_data - min_coord) / (max_coord - min_coord)
    
    block_data = []
    block_spatial = []

    for i in range(n_blocks_x):
        for j in range(n_blocks_y):
            min_x, max_x = i / n_blocks_x, (i + 1) / n_blocks_x
            min_y, max_y = j / n_blocks_y, (j + 1) / n_blocks_y
            
            
            block_indices = np.where(
                (norm_spatial[:, 0] >= min_x) & (norm_spatial[:, 0] < max_x) &
                (norm_spatial[:, 1] >= min_y) & (norm_spatial[:, 1] < max_y)
            )[0]
            
            if len(block_indices) > 0:
                block_data.append(adata.X[block_indices].sum(axis=0))
                block_spatial.append(spatial_data[block_indices].mean(axis=0))
    
    block_data = np.array(block_data)
    block_spatial = np.array(block_spatial)

    # Plot the original data with the downsampled grid overlay
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=800)

    # Generate the color list from the 'tab20' colormap
    

    # Scatter plot for each cell type in the original codex_adata
    for j, i in enumerate(adata.obs['type'].unique()):
        ax.scatter(norm_spatial[adata.obs['type'] == i, 0],
                   norm_spatial[adata.obs['type'] == i, 1],
                   s=size, color=color_list[j], label=i)

    # Add a legend
    #plt.legend(adata.obs['type'].unique(), bbox_to_anchor=(1.01, 1))

    # Calculate grid dimensions based on normalized coordinates
    x_step = 1 / n_blocks_x
    y_step = 1 / n_blocks_y

    # Draw the grid lines with specified linewidth
    for i in range(n_blocks_x + 1):
        # Vertical lines
        ax.plot([i * x_step, i * x_step], [0, 1], linestyle='--', color='lightcoral', alpha=0.8, linewidth=0.7)
    
    for j in range(n_blocks_y + 1):
        # Horizontal lines
        ax.plot([0, 1], [j * y_step, j * y_step], linestyle='--', color='lightcoral', alpha=0.8, linewidth=0.7)

    # Invert y-axis to match spatial data
    if invert_yaxis:
        plt.gca().invert_yaxis()
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    # Display the plot
    plt.show()

    return block_data, block_spatial

def downsample_and_plot_spatial_data_smooth(adata, color_list, n_blocks_x=5, n_blocks_y=5,size=15, figsize=(3,6),invert_yaxis=True):
    sigma=0.8
    spatial_data = adata.obsm['spatial']
    min_coord = spatial_data.min(axis=0)
    max_coord = spatial_data.max(axis=0)
    norm_spatial = (spatial_data - min_coord) / (max_coord - min_coord)
    
    block_data = []
    block_spatial = []

    for i in range(n_blocks_x):
        for j in range(n_blocks_y):
            min_x, max_x = i / n_blocks_x, (i + 1) / n_blocks_x
            min_y, max_y = j / n_blocks_y, (j + 1) / n_blocks_y
            
            block_center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
            
            
            block_indices = np.where(
                (norm_spatial[:, 0] >= min_x) & (norm_spatial[:, 0] < max_x) &
                (norm_spatial[:, 1] >= min_y) & (norm_spatial[:, 1] < max_y)
            )[0]
            
            if len(block_indices) > 0:
                # Compute distances and apply Gaussian weights based on distance from block center
                distances = np.linalg.norm(norm_spatial[block_indices] - block_center, axis=1)
                weights = multivariate_normal.pdf(distances, mean=0, cov=sigma**2)
                
                
                
                # Check if weighted_data and weighted_spatial are consistent in shape
                weighted_data = (adata.X[block_indices].T * 1).T.sum(axis=0) / weights.sum()
                
                #if weights.shape
                block_data.append(weighted_data)
                
                block_spatial.append(spatial_data[block_indices].mean(axis=0))
                           
    block_data = np.array(block_data)
    block_spatial = np.array(block_spatial)

    # Plot the original data with the downsampled grid overlay
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=800)

    # Generate the color list from the 'tab20' colormap
    

    # Scatter plot for each cell type in the original codex_adata
    for j, i in enumerate(adata.obs['type'].unique()):
        ax.scatter(norm_spatial[adata.obs['type'] == i, 0],
                   norm_spatial[adata.obs['type'] == i, 1],
                   s=size, color=color_list[j], label=i)

    # Add a legend
    #plt.legend(adata.obs['type'].unique(), bbox_to_anchor=(1.01, 1))

    # Calculate grid dimensions based on normalized coordinates
    x_step = 1 / n_blocks_x
    y_step = 1 / n_blocks_y

    # Draw the grid lines with specified linewidth
    for i in range(n_blocks_x + 1):
        # Vertical lines
        ax.plot([i * x_step, i * x_step], [0, 1], linestyle='--', color='lightcoral', alpha=0.8, linewidth=0.7)
    
    for j in range(n_blocks_y + 1):
        # Horizontal lines
        ax.plot([0, 1], [j * y_step, j * y_step], linestyle='--', color='lightcoral', alpha=0.8, linewidth=0.7)

    # Invert y-axis to match spatial data
    if invert_yaxis:
        plt.gca().invert_yaxis()
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    # Display the plot
    plt.show()

    return block_data, block_spatial


def plot_pca_cluster(adata,
                     res=0.1,
                     size=15,
                     color_list = [
                            '#f94144',
                            '#f3722c',
                            '#f8961e',
                            '#f9c74f',
                            '#90be6d',
                            '#43aa8b',
                            '#577590',  
                            '#226CE0',
                            '#534B62'
                                         
                     ],
                     figsize=(3,6),
                     legd = False,
                     return_=False,
                     invert_yaxis=True,
                     axis_ = True,
                    ):
    

    adata_ = adata.copy()
    sc.pp.normalize_total(adata_, target_sum=1e4)
    sc.pp.log1p(adata_)
    sc.tl.pca(adata_, svd_solver='arpack')
    sc.pp.neighbors(adata_,n_neighbors=30,use_rep='X_pca')
    sc.tl.leiden(adata_, resolution=res,random_state=42)

    adata_.obs['cluster'] = adata_.obs['leiden']#.map(map_leiden)
    fig, ax = plt.subplots(1,1,figsize=figsize,dpi=800)
    for j,i in enumerate(adata_.obs['cluster'].unique()):
        print(i)
        plt.scatter(adata_.obsm['spatial'][adata_.obs['cluster']==i,0],
                    adata_.obsm['spatial'][adata_.obs['cluster']==i,1],
                    s=size,
                    color = color_list[j]
                   )
    if legd:
        plt.legend(adata_.obs['cluster'].unique(),bbox_to_anchor=(1.01,0.8))
    if invert_yaxis:
        plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])  
    if not axis_:
        plt.axis('off')
    if return_:
        return adata_
    

def plot_latent_cluster(adata_,res=0.1,size=15,
                        use_rep = 'X_pca',
                        color_list = [
                            '#f94144',
                            '#f3722c',
                            '#f8961e',
                            '#f9c74f',
                            '#90be6d',
                            '#43aa8b',
                            '#577590',  
                            '#226CE0',
                            '#77479F'
                                         
                     ],
                     figsize=(3,6),
                     legd = False,
                     return_=False,
                     invert_yaxis=True,
                     bbox_to_anchor=(1.01,0.8),
                     legend_fontsize = 10,
                     legend_markerscale=5
                       ):
    
    

    sc.pp.neighbors(adata_,n_neighbors=64,use_rep=use_rep)
    sc.tl.leiden(adata_, resolution=res)
    
    #adata_.obs['cluster'] = adata_.obs['leiden'].map(map_leiden)
    adata_.obs['cluster'] = adata_.obs['leiden'].astype(str)
    fig, ax = plt.subplots(1,1,figsize=figsize,dpi=800)
    for j,i in enumerate(sorted(adata_.obs['cluster'].unique())):
        
        plt.scatter(adata_.obsm['spatial'][adata_.obs['cluster']==i,0],
                    adata_.obsm['spatial'][adata_.obs['cluster']==i,1],
                    s=size,
                    color = color_list[j]
                   )
    if legd:
        plt.legend(sorted(adata_.obs['cluster'].unique()),
                   bbox_to_anchor=bbox_to_anchor,
                   prop={'size': legend_fontsize} ,
                   markerscale=legend_markerscale
                  )
    
                   
    if invert_yaxis:
        plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])  
    plt.axis('off')
    
    if return_:
        return adata_
    
    
def visualize_subgraph(data):
    
    plt.figure(figsize = (5,4),dpi=800)
    # Convert to NetworkX graph with node attributes
    G = to_networkx(data, node_attrs=['x', 'spatial_coords', 'center_cell'])

    # Extract spatial coordinates for positioning nodes
    pos = {i: data.spatial_coords[i].cpu().numpy() for i in range(data.num_nodes)}

    # Extract the center cell index
    center_cell_idx = data.center_cell.nonzero(as_tuple=True)[0].item()

    # Define node colors
    node_colors = ['red' if i == center_cell_idx else 'skyblue' for i in range(data.num_nodes)]

    # Draw the graph
    nx.draw(G, pos, with_labels=False, node_size=700, node_color=node_colors, font_size=10,alpha=1, font_color='k', edge_color='k')

    # Show the plot
    plt.show()
    
    
    
def visualize_full_dataset_with_subgraph(dataloader):
    all_coords = []

    # Step 1: Scatter Plot for All Nodes
    for batch in dataloader:
        for i in range(len(batch)):
            data = batch[i]

            # Extract spatial coordinates
            coords = data.spatial_coords.cpu().numpy()
            all_coords.append(coords)

    # Combine all coordinates into a single array
    all_coords = np.vstack(all_coords)

    # Step 2: Prepare Subgraph for the First Batch
    data = next(iter(dataloader))[0]  # Get the first batch

    # Convert to NetworkX graph with node attributes
    G = to_networkx(data, node_attrs=['x', 'spatial_coords', 'center_cell'])

    # Extract spatial coordinates for positioning nodes
    pos = {i: data.spatial_coords[i].cpu().numpy() for i in range(data.num_nodes)}

    # Extract the center cell index
    center_cell_idx = data.center_cell.nonzero(as_tuple=True)[0].item()

    # Define node colors for the subgraph
    node_colors = ['red' if i == center_cell_idx else 'skyblue' for i in range(data.num_nodes)]

    # Draw the combined plot
    plt.figure(figsize=(6,14),dpi=800)
    
    # Scatter plot of all nodes
    plt.scatter(all_coords[:, 0], all_coords[:, 1], c='lightgray', s=100, alpha=0.6, label='All Nodes')
    
    # Draw the subgraph on top of the scatter plot
    sub_nodes = list(G.neighbors(center_cell_idx)) + [center_cell_idx]
    sub_pos = {i: pos[i] for i in sub_nodes}
    nx.draw(G.subgraph(sub_nodes), sub_pos, with_labels=False, node_size=50, node_color=node_colors, font_size=0, font_color='k', edge_color='k', ax=plt.gca())

    #plt.title("Combined Scatter Plot and Subgraph")
    
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
