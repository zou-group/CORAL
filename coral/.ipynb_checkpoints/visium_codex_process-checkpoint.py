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

from sklearn.neighbors import KDTree,NearestNeighbors
from coral import coral_main, VisCoxDataset, utils, utils_exp, utils_simu
from scipy.spatial import cKDTree

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


def preprocess_visium(visium_data,
                      used_gene=None,
                      lognorm=False,
                      select_gene= None,
                     ):
    """
    get the n top genes, and included codex related genes
    """
    #print(len(selected_gene))
    visium_data.var_names_make_unique()
    
    ## delete MT genes and RPL/RPS gene
    visium_data = visium_data[:,[not ((i.startswith('MT-'))|(i.startswith('RPL'))|(i.startswith('RPS'))) for i in visium_data.to_df().columns]]
    
    if lognorm:
        sc.pp.normalize_total(visium_data,target_sum=1e4,inplace=True)
        sc.pp.log1p(visium_data)
    
    print(len(select_gene))
    print(visium_data)
    visium_data = visium_data[:,select_gene]
    #sc.pp.highly_variable_genes(visium_data, n_top_genes=used_gene)
    print(visium_data.shape)
    return visium_data

def robust_z_scale(df, columns):
    
    scaled_df = df.copy()  
    for column in columns:
        median = np.median(df[column])
        q75, q25 = np.percentile(df[column], [90, 32])
        iqr = q75 - q25
        scaled_df[column] = (df[column] - median) / iqr
    
    return scaled_df

def asinh_transform(x, cofactor=5):
    return np.arcsinh(x / (cofactor * x.quantile(0.2)))


def preprocess_codex(codex_data):
    codex_data_normed = asinh_transform(codex_data)
    #codex_data_normed = robust_z_scale(codex_data_normed, codex_data_normed.columns)
    codex_data_normed.fillna(0, inplace=True)
    codex_data_normed.replace([np.inf, -np.inf], 0, inplace=True)

    return codex_data_normed



def load_data(visium_path, trans_loc, codex_path, type_path, cell_type_label_mapping, used_gene = 2000,select_gene=None):
    
    visium_adata = sc.read_visium(visium_path)
    visium_adata.var_names_make_unique()
    transformed_loc = pd.read_csv(trans_loc,index_col=0)
    transformed_loc.index = transformed_loc['barcode']
    transformed_loc = transformed_loc.loc[visium_adata.obs_names,:]
    visium_adata.obs = transformed_loc
    visium_adata = preprocess_visium(visium_adata, used_gene=used_gene,lognorm=False,select_gene=select_gene)

    codex_df = pd.read_csv(codex_path,index_col=0)
    
    codex_adata = anndata.AnnData(codex_df.iloc[:,3:])
    codex_adata.obs = codex_df.iloc[:,:3]
    codex_adata.obs.index = codex_adata.obs['cell_id'].values.astype(str)
    codex_adata = codex_adata[list(codex_adata.obs[['cell_id']].sort_values(by='cell_id').index)]
    
    cell_type = pd.read_csv(type_path,index_col=0)
    cell_type.index = cell_type.index.astype(str)
    
    codex_adata.obs = pd.concat([codex_adata.obs,
             cell_type[['CELL_TYPE']]]
             ,axis=1
            ).dropna(subset=['CELL_TYPE'])

    codex_adata.obs['CELL_TYPE'] =codex_adata.obs['CELL_TYPE'].map(cell_type_label_mapping)
    codex_adata = codex_adata[cell_type.index]
    
    codex_adata.X = preprocess_codex(codex_adata.to_df())
    
    return visium_adata, codex_adata

