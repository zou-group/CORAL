import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
from scipy.spatial import cKDTree
import pandas as pd
import torch
from torch_geometric.data import Data, DataLoader
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import issparse


def add_cluster(
    adata,
    res: float = 0.1,
    use_rep_for_cluster: str | None = None,
    need_lognormed: bool | None = None,
    n_neighbors: int = 100, 
):

    # Copy input to avoid modifying the original AnnData
    adata_ = adata.copy()

    if need_lognormed:
        sc.pp.normalize_total(adata_, target_sum=1e4)
        sc.pp.log1p(adata_)

    
    if use_rep_for_cluster=='X_pca':
        sc.tl.pca(adata_, svd_solver="arpack")
        use_rep_for_cluster = "X_pca"

    # Compute neighbors and Leiden clusters
    sc.pp.neighbors(adata_, n_neighbors=n_neighbors, use_rep=use_rep_for_cluster)
    sc.tl.leiden(adata_, resolution=res, random_state=0, flavor="igraph")

    adata_.obs["cluster"] = adata_.obs["leiden"].astype(str)

    adata.obs['cluster'] = adata_.obs["cluster"]
    return adata



def preprocess_data(adata_smoothed, adata_adt):
    visium_expr = adata_smoothed.X.toarray() if issparse(adata_smoothed.X) else np.array(adata_smoothed.X)
    
    visium_coords = adata_smoothed.obsm['spatial']

    
    codex_expr = adata_adt.X.toarray() if issparse(adata_adt.X) else np.array(adata_adt.X)
    
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
    #print(codex_expr.shape)
    #print(codex_to_visium_expr.shape)
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

def downsample_spatial_data(adata, n_blocks_x=12, n_blocks_y=12):

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
                (norm_spatial[:, 0] >= min_x) & (norm_spatial[:, 0] <= max_x) &
                (norm_spatial[:, 1] >= min_y) & (norm_spatial[:, 1] <= max_y)
            )[0]
            
            if len(block_indices) > 0:
                block_data.append(adata.X[block_indices].sum(axis=0))
                block_spatial.append(spatial_data[block_indices].mean(axis=0))

    block_spatial = np.array(block_spatial)
    block_data = np.array(block_data)
    adata_downsampled = sc.AnnData(X=block_data)
    adata_downsampled.obsm['spatial'] = block_spatial
        
    return adata_downsampled

