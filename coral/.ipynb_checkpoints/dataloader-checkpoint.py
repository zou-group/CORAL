import numpy as np
import pandas as pd
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from sklearn.neighbors import KDTree
from torch_geometric.data import Batch
import scanpy as sc

def preprocess_data(adata_smoothed, adata_adt):
    visium_expr = adata_smoothed.X
    visium_coords = adata_smoothed.obsm['spatial']

    codex_expr = adata_adt.X
    codex_coords = adata_adt.obsm['spatial']


    # Map each CODEX cell to the nearest Visium spot
    tree = cKDTree(visium_coords)
    _, indices = tree.query(codex_coords)
    codex_to_visium_expr = visium_expr[indices]

    # Combine mapped Visium data with CODEX data
    combined_expr = np.concatenate([codex_to_visium_expr, codex_expr], axis=1)
    return combined_expr, codex_coords


class VisCoxDataset(Dataset):
    """
    VisiumDataset
    """

    def __init__(
        self,
        visium_data,
        codex_data,
        sc_data = None,
        window=100,
        transform=None, 
        sample_id=None,
    ):

        codex_data.obs['sample_id']  = sample_id
        x = visium_data.X if isinstance(visium_data.X, np.ndarray) else visium_data.X.A
        y = codex_data.X if isinstance(codex_data.X, np.ndarray) else codex_data.X.A
        
        self.expr_mat    = pd.DataFrame(x, index=visium_data.obs_names, columns=visium_data.var_names)
        self.protein_mat = pd.DataFrame(y, index=codex_data.obs_names,  columns=codex_data.var_names)
        
        if not sc_data is None:
            self.sc_rna = sc_data.X if isinstance(sc_data.X, np.ndarray) else sc_data.X.A
        else:
            self.sc_rna=None
        
        if 'CELL_TYPE' in codex_data.obs.columns:
            self.cell_type = codex_data.obs['CELL_TYPE']
        else:
            
            self.cell_type = None
        
        self.spot_loc = visium_data.obsm['spatial']#[['transformed coords x','transformed coords y']]
        self.cell_loc = codex_data.obsm['spatial']#[['x','y']]
        self.window = window
        self.cell_id = codex_data.obs[['cell_id']]
        self.sample_id = codex_data.obs[['sample_id']]
        

        # KDTree for efficient spatial indexing
        self.tree = KDTree(self.cell_loc, leaf_size=10, metric='euclidean')
        self.transform = transform 
        self.neighbors = codex_data.uns['spatial_neighbors']
    

        
    def __len__(self):
        return len(self.expr_mat)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        spot_sampled = np.array(self.spot_loc[idx, :], dtype='float')
        indices = self.tree.query_radius(spot_sampled.reshape(1, -1), r=self.window)
        spot_id = np.array(list(range(self.spot_loc.shape[0]))[idx], dtype='float')
        
        if len(indices[0]) == 0:
            logging.warning(f"No neighboring cells found for sample {idx}. Returning placeholder.")
            codex_sampled = torch.Tensor() 
            cell_type = torch.Tensor() 
            cell_id = torch.Tensor() 
            sample_id = torch.Tensor() 
            cell_loc = torch.Tensor() 
            cell_type = torch.Tensor() 
            sc_rna = torch.Tensor()
        else:
            codex_sampled = torch.Tensor(np.array(self.protein_mat.iloc[indices[0], :], dtype='float'))
            cell_loc = torch.Tensor(np.array(self.cell_loc[indices[0],:], dtype='float'))
            cell_id = torch.Tensor(np.array(self.cell_id.iloc[indices[0],:], dtype='float'))
            sample_id = torch.Tensor(np.array(self.sample_id.iloc[indices[0],:], dtype='float'))
            neighbors = np.array(self.neighbors.iloc[indices[0],:], dtype='int')
            
            neighbors_node = None
            edge_indices = None
            
            neighbors_node, edge_indices = create_cell_graph(self.protein_mat, neighbors)
            graph_data_list = []
            for i in range(len(neighbors_node)):
                graph_data = Data(x=neighbors_node[i], edge_index=edge_indices[i])
                graph_data_list.append(graph_data)
            if self.cell_type is None:
                cell_type = None
            else:
                cell_type =torch.Tensor(np.array(self.cell_type.iloc[indices[0]], dtype='int'))

            if self.sc_rna is None:
                sc_rna=None 
            else:
                sc_rna = torch.Tensor(np.array(self.sc_rna[indices[0], :], dtype='float'))
            
        
        

        sample = {'visium'    : torch.Tensor(np.array(self.expr_mat.iloc[idx, :], dtype='float')), 
                  'codex'     : codex_sampled,
                  'cell_type' : cell_type,
                  'spot_loc'   : torch.Tensor(np.array(self.spot_loc[idx,:], dtype='float')),
                  'spot_id': spot_id,
                  'cell_loc'  : cell_loc,
                  'sc_rna'    : sc_rna,
                  'cell_id'   : cell_id,
                  'sample_id' : sample_id,
                  'neighbors'  : neighbors,
                  'neighbors_node':neighbors_node,
                  'edge_indices':edge_indices,
                  'graph_data':graph_data_list
                  }
        
        if self.transform:
            sample = self.transform(sample)
        return sample
       
    @staticmethod
    def collate_fn(batch):
        visium = [item['visium'] for item in batch]
        codex = [item['codex'] for item in batch]
        cell_type = [item['cell_type'] for item in batch]
        spot_loc = [item['spot_loc'] for item in batch]
        spot_id = [item['spot_id'] for item in batch]
        cell_loc = [item['cell_loc'] for item in batch]
        cell_id = [item['cell_id'] for item in batch]
        sc_rna = [item['sc_rna'] for item in batch]
        sample_id = [item['sample_id'] for item in batch]
        neighbors = [item['neighbors'] for item in batch]
        graph_data = [item['graph_data'] for item in batch]
        return {'visium': visium, 
                'codex': codex,
                'cell_type':cell_type,
                'spot_loc':spot_loc,
                'spot_id':spot_id,
                'cell_loc':cell_loc,
                'sc_rna':sc_rna,
                'cell_id':cell_id,
                'sample_id':sample_id,
                'neighbors':neighbors,
                'graph_data':graph_data,
               }
            


