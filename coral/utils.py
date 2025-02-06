import scanpy as sc
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import scipy
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
import torch
import anndata
import sys
sys.path.append('../coral')
import utils_exp
from scipy.spatial import cKDTree
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import mean_squared_error


class DataWithCellType(Data):
    def __init__(self, x=None, edge_index=None, cell_type=None, **kwargs):
        super(DataWithCellType, self).__init__(x=x, edge_index=edge_index, **kwargs)
        self.cell_type = cell_type


def preprocess_data(adata_smoothed, adata_adt):
    visium_expr = adata_smoothed.X
    visium_coords = adata_smoothed.obsm['spatial']

    codex_expr = adata_adt.X
    codex_coords = adata_adt.obsm['spatial']

    # Normalize the data
    #visium_expr = (visium_expr - visium_expr.mean(axis=0)) / visium_expr.std(axis=0)
    #codex_expr = (codex_expr - codex_expr.mean(axis=0)) / codex_expr.std(axis=0)

    # Map each CODEX cell to the nearest Visium spot
    tree = cKDTree(visium_coords)
    _, indices = tree.query(codex_coords)
    codex_to_visium_expr = visium_expr[indices]
    
    assert indices.max() < visium_coords.shape[0], "Index out of range"


    # Combine mapped Visium data with CODEX data
    #print(codex_to_visium_expr)
    combined_expr = np.concatenate([codex_to_visium_expr, codex_expr], axis=1)

    # One-hot encode cell type information specific to CODEX data
    cell_types = adata_adt.obs['cell_type'].astype('category').cat.codes
    one_hot_cell_types = pd.get_dummies(cell_types).values
    
    return combined_expr, codex_coords, one_hot_cell_types, indices, visium_expr



def prepare_local_subgraphs(combined_expr, codex_coords, one_hot_cell_types, spot_indices, visium_expr, n_neighbors=20):


    features = torch.tensor(combined_expr, dtype=torch.float32)
    cell_types = torch.tensor(one_hot_cell_types, dtype=torch.float32)
    spot_indices = torch.tensor(spot_indices, dtype=torch.long)
    spatial_coords = torch.tensor(codex_coords, dtype=torch.float32)

    
    adjacency_matrix = kneighbors_graph(codex_coords, n_neighbors=n_neighbors, mode='connectivity', include_self=True)

    data_list = []
    for i in range(codex_coords.shape[0]):
        neighbors = set(adjacency_matrix[i].nonzero()[1])
        
        # Ensure all cells belonging to the same Visium spot are included
        spot_neighbors = set(np.where(spot_indices == spot_indices[i])[0])
        neighbors.update(spot_neighbors)
        
        neighbors = list(neighbors)
    
        # Adjust the indices to match subgraph's node indices
        neighbor_indices = {j: idx for idx, j in enumerate(neighbors)}
        
        
        edge_index = []
        global_edge_ids = []  # To store global ID pairs for edges
        
        for neighbor in neighbors:
            if neighbor != i:  # Only add edges between the central node and its neighbors
                edge_index.append([neighbor_indices[i], neighbor_indices[neighbor]])
                edge_index.append([neighbor_indices[neighbor], neighbor_indices[i]])
                global_edge_ids.append([i, neighbor])
                global_edge_ids.append([neighbor,i])
                
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        global_edge_ids = torch.tensor(global_edge_ids, dtype=torch.long)  # Convert to tensor and transpose

        center_flag = torch.zeros(len(neighbors), dtype=torch.bool)
        center_flag[neighbor_indices[i]] = True  # Mark the correct cell as the center cell


        subgraph = Data(
            x=features[neighbors],
            edge_index=edge_index,
            global_edge_ids=global_edge_ids,
            cell_type=cell_types[neighbors],
            spot_indices=spot_indices[neighbors],
            spatial_coords=spatial_coords[neighbors],
            visium_spot=torch.tensor([spot_indices[i]], dtype=torch.long),  # Record the Visium index of the center cell
            visium_spot_exp=torch.tensor(visium_expr[spot_indices[i]], dtype=torch.float32).unsqueeze(0),  # Record true expression values
            center_cell=center_flag,
            
        )
        data_list.append(subgraph)

    dataloader = DataLoader(data_list, batch_size=8, shuffle=True)
    
    return dataloader




def reconstruction_accuracy(original_data, generated_data):
    mse = mean_squared_error(original_data, generated_data)
    rmse = np.sqrt(mse)
    return mse, rmse




def display_reconst(df_true,
                    df_pred,
                    density=False,
                    marker_genes=None,
                    sample_rate=0.01,
                    size=(4, 4),
                    spot_size=1,
                    title=None,
                    x_label='',
                    y_label='',
                    min_val=None,
                    max_val=None,
                    ):
    """
    Scatter plot - raw gexp vs. reconstructed gexp
    """
    assert 0 < sample_rate <= 1, \
        "Invalid downsampling rate for reconstruct scatter plot: {}".format(sample_rate)

    if marker_genes is not None:
        marker_genes = set(marker_genes)

    df_true_sample = df_true.sample(frac=sample_rate, random_state=0)
    df_pred_sample = df_pred.loc[df_true_sample.index]

    plt.rcParams["figure.figsize"] = size
    plt.figure(dpi=800)
    ax = plt.gca()

    xx = df_true_sample.T.to_numpy().flatten()
    yy = df_pred_sample.T.to_numpy().flatten()

    if density:
        for gene in df_true_sample.columns:
            try:
                gene_true = df_true_sample[gene].values
                gene_pred = df_pred_sample[gene].values
                gexp_stacked = np.vstack([df_true_sample[gene].values, df_pred_sample[gene].values])

                z = gaussian_kde(gexp_stacked)(gexp_stacked)
                ax.scatter(gene_true, gene_pred, c=z, s=spot_size, alpha=0.5)
            except np.linalg.LinAlgError as e:
                pass

    elif marker_genes is not None:
        color_dict = {True: 'red', False: 'green'}
        gene_colors = np.vectorize(
            lambda x: color_dict[x in marker_genes]
        )(df_true_sample.columns)
        colors = np.repeat(gene_colors, df_true_sample.shape[0])

        ax.scatter(xx, yy, c=colors, s=spot_size, alpha=0.5)

    else:
        ax.scatter(xx, yy, s=spot_size, alpha=0.5)

    ax.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=1, label='y = x')

    #min_val = min(xx.min(), yy.min())
    #max_val = max(xx.max(), yy.max())
    #ax.set_xlim(min_val, 400)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    #ax.set_ylim(min_val, 400)

    plt.suptitle(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.show()
    
    
    
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt

def visualize_subgraph(data):
    # Convert to NetworkX graph with node attributes
    G = to_networkx(data, node_attrs=['x', 'spatial_coords', 'center_cell'])

    # Extract spatial coordinates for positioning nodes
    pos = {i: data.spatial_coords[i].cpu().numpy() for i in range(data.num_nodes)}

    # Extract the center cell index
    center_cell_idx = data.center_cell.nonzero(as_tuple=True)[0].item()

    # Define node colors
    node_colors = ['red' if i == center_cell_idx else 'skyblue' for i in range(data.num_nodes)]

    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_size=700, node_color=node_colors, font_size=10, font_color='black', edge_color='gray')

    # Show the plot
    plt.show()
    
    
    
    
def cluster_latent_rep(latent_rep, n_clusters=None, resolution=1.0):
    # Create an AnnData object from the latent representation
    adata = anndata.AnnData(X=latent_rep)
    
    # Compute the neighborhood graph
    sc.pp.neighbors(adata, n_neighbors=64, use_rep='X')  # You can adjust n_neighbors based on your data
    
    # Perform Leiden clustering
    sc.tl.leiden(adata, resolution=resolution)
    
    # Extract the cluster labels
    clusters = adata.obs['leiden'].astype(int).values
    
    return clusters


def reindex_adata_qz(adata: anndata.AnnData, adata_qz: anndata.AnnData) -> anndata.AnnData:
    """
    Reindex adata_qz to match the order of adata based on spatial coordinates.

    Parameters:
    - adata: AnnData object containing the reference spatial coordinates.
    - adata_qz: AnnData object to be reindexed based on spatial coordinates.

    Returns:
    - adata_qz_reindexed: Reindexed AnnData object.
    """

    # Extract spatial coordinates
    spatial_coords_adata = adata.obsm['spatial']
    spatial_coords_adata_qz = adata_qz.obsm['spatial']

    # Find nearest neighbors
    tree = cKDTree(spatial_coords_adata_qz)
    distances, indices = tree.query(spatial_coords_adata)

    # Reindex adata_qz
    adata_qz_reindexed = adata_qz[indices].copy()
    adata_qz_reindexed.obs.index = adata.obs.index

    return adata_qz_reindexed
