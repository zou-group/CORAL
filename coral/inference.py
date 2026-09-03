import torch
import numpy as np
import anndata
from scipy.spatial import cKDTree

from collections import defaultdict

def average_attention_weights_for_unique_edges(edges_all, attn_weights_all):
    """
    Average attention weights for unique edges.
    
    Parameters:
    - edges_all: numpy array of shape (2, num_edges) representing all edge indices.
    - attn_weights_all: numpy array of shape (num_edges, num_heads) representing attention weights.
    
    Returns:
    - unique_edges: numpy array of shape (2, num_unique_edges) with unique edge indices.
    - averaged_weights: numpy array of shape (num_unique_edges,) with averaged attention weights.
    """
    edge_dict = defaultdict(list)
    
    # Convert edges to tuples to handle undirected edges (min, max)
    for i, (src, dst) in enumerate(edges_all.T):
        edge = tuple(sorted((src, dst)))
        edge_dict[edge].append(np.mean(attn_weights_all[i]))
    
    # Compute average attention weights for unique edges
    unique_edges = np.array(list(edge_dict.keys())).T
    averaged_weights = np.array([np.mean(weights) for weights in edge_dict.values()])
    
    return unique_edges, averaged_weights

    
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


def generate_and_validate(model, dataloader,device, hires_adata, ground_truth = False):
    model.eval()
    generated_expr = []
    generated_protein = []
    
    latent_rep = []
    locations = []
    visium_true = []
    codex_true = []
    attn_weights_all = []
    edge_indices_all = []
    
    v_values = []
    cell_types = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            edge_index = batch.global_edge_ids.t()
            spatial_coords = batch.spatial_coords

        
            output = model(batch, device)

            
            center_cell_idx = batch.center_cell.nonzero(as_tuple=True)[0]
            

            
            generated_expr.append(output['px_rate_aggregated'].cpu().numpy())
            generated_protein.append(output['py_rate'].cpu().numpy())
            latent_rep.append(output['z_mu'].cpu().numpy())
            if ground_truth:
                visium_true.append(batch.visium_spot_exp.cpu().numpy())
                codex_true.append(batch.x[:, model.visium_dim:][center_cell_idx].cpu().numpy())
            attn_weights_all.append(output['attn_weights_2'][1].cpu().numpy())
            edge_indices_all.append(edge_index.cpu().numpy())
            
            v_values.append(output['v'].cpu().numpy())
            cell_types.append(batch.cell_type[center_cell_idx].cpu().numpy())
            locations.append(batch.spatial_coords[center_cell_idx].cpu().numpy())
    
    
    
    generated_expr = np.concatenate(generated_expr, axis=0)
    generated_protein = np.concatenate(generated_protein, axis=0)
    latent_rep = np.concatenate(latent_rep, axis=0)
    locations = np.concatenate(locations, axis=0)
    if ground_truth:
        visium_true = np.concatenate(visium_true, axis=0)
        codex_true = np.concatenate(codex_true, axis=0)
    v_values = np.concatenate(v_values, axis=0)
    cell_types = np.concatenate(cell_types, axis=0)
    
    edges_all = np.concatenate(edge_indices_all, axis=1)
    attn_weights_all = np.concatenate(attn_weights_all, axis=0)
    

    
    adata_model_gener = anndata.AnnData(generated_protein)
    adata_model_gener.obsm['generated_expr'] = generated_expr
    adata_model_gener.obsm['coral'] = latent_rep
    adata_model_gener.obsm['spatial'] = locations
    adata_model_gener.obsm['v_values'] = v_values
    adata_model_gener.obsm['cell_types'] = cell_types
    adata_model_gener.obsm['cell_types'] = cell_types
    if ground_truth:
        adata_model_gener.obsm['visium_true'] = visium_true
        adata_model_gener.obsm['codex_true'] = codex_true        
    adata_model_gener_reindexed = reindex_adata_qz(hires_adata, adata_model_gener)

    
    return adata_model_gener_reindexed, edges_all, attn_weights_all