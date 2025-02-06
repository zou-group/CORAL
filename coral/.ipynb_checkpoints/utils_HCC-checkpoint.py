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

meta_info =[['71_2_pre', 'FinalLiv-27_c001_v001_r001_reg005_he_aligned.tif', '071-2.ome.tif','cytassist_71_pre', 'FinalLiv-27_c001_v001_r001_reg005_codex.csv','071_2_tissue_positions_list.csv','FinalLiv-27_c001_v001_r001_reg005.cell_types.csv'],
            ['71_3_pre', 'FinalLiv-27_c001_v001_r001_reg006_he_aligned.tif', '071-3.ome.tif','cytassist_71_pre', 'FinalLiv-27_c001_v001_r001_reg006_codex.csv','071_3_tissue_positions_list.csv','FinalLiv-27_c001_v001_r001_reg006.cell_types.csv'],
              
            ['72_2_pre', 'FinalLiv-27_c001_v001_r001_reg009_he_aligned.tif', '072-2.ome.tif','cytassist_72_pre', 'FinalLiv-27_c001_v001_r001_reg009_codex.csv','072_2_tissue_positions_list.csv','FinalLiv-27_c001_v001_r001_reg009.cell_types.csv'],
            ['72_3_pre', 'FinalLiv-27_c001_v001_r001_reg010_he_aligned.tif', '072-3.ome.tif','cytassist_72_pre', 'FinalLiv-27_c001_v001_r001_reg010_codex.csv','072_3_tissue_positions_list.csv','FinalLiv-27_c001_v001_r001_reg010.cell_types.csv'],
            ['72_4_post','FinalLiv-27_c001_v001_r001_reg011_he_aligned.tif', '072-4.ome.tif','cytassist_72_post','FinalLiv-27_c001_v001_r001_reg011_codex.csv','072_4_tissue_positions_list.csv','FinalLiv-27_c001_v001_r001_reg011.cell_types.csv'],
            ['72_5_post','FinalLiv-27_c001_v001_r001_reg012_he_aligned.tif', '072-5.ome.tif','cytassist_72_post','FinalLiv-27_c001_v001_r001_reg012_codex.csv','072_5_tissue_positions_list.csv','FinalLiv-27_c001_v001_r001_reg012.cell_types.csv'],
            
            ['73_2_pre', 'FinalLiv-27_c001_v001_r001_reg013_he_aligned.tif', '073-2.ome.tif','cytassist_73_pre', 'FinalLiv-27_c001_v001_r001_reg013_codex.csv','073_2_tissue_positions_list.csv','FinalLiv-27_c001_v001_r001_reg013.cell_types.csv'],
            ['73_3_pre', 'FinalLiv-27_c001_v001_r001_reg014_he_aligned.tif', '073-3.ome.tif','cytassist_73_pre', 'FinalLiv-27_c001_v001_r001_reg014_codex.csv','073_3_tissue_positions_list.csv','FinalLiv-27_c001_v001_r001_reg014.cell_types.csv'],
            #['73_4_pre', 'FinalLiv-27_c001_v001_r001_reg015_he_aligned.tif', '073-4.ome.tif','cytassist_73_post','FinalLiv-27_c001_v001_r001_reg015_codex.csv','073_4_tissue_positions_list.csv','FinalLiv-27_c001_v001_r001_reg015.cell_types.csv'],
            ['73_5_post','FinalLiv-27_c001_v001_r001_reg016_he_aligned.tif', '073-5.ome.tif','cytassist_73_post','FinalLiv-27_c001_v001_r001_reg016_codex.csv','073_5_tissue_positions_list.csv','FinalLiv-27_c001_v001_r001_reg016.cell_types.csv'],
            
            ['74_2_pre', 'FinalLiv-27_c001_v001_r001_reg017_he_aligned.tif', '074-2.ome.tif','cytassist_74_pre', 'FinalLiv-27_c001_v001_r001_reg017_codex.csv','074_2_tissue_positions_list.csv','FinalLiv-27_c001_v001_r001_reg017.cell_types.csv'],
            ['74_3_pre', 'FinalLiv-27_c001_v001_r001_reg018_he_aligned.tif', '074-3.ome.tif','cytassist_74_pre', 'FinalLiv-27_c001_v001_r001_reg018_codex.csv','074_3_tissue_positions_list.csv','FinalLiv-27_c001_v001_r001_reg018.cell_types.csv'],
            ['74_4_post','FinalLiv-27_c001_v001_r001_reg019_he_aligned.tif', '074-4.ome.tif','cytassist_74_post','FinalLiv-27_c001_v001_r001_reg019_codex.csv','074_4_tissue_positions_list.csv','FinalLiv-27_c001_v001_r001_reg019.cell_types.csv'],
            ['74_5_post','FinalLiv-27_c001_v001_r001_reg020_he_aligned.tif', '074-5.ome.tif','cytassist_74_post','FinalLiv-27_c001_v001_r001_reg020_codex.csv','074_5_tissue_positions_list.csv','FinalLiv-27_c001_v001_r001_reg020.cell_types.csv'],
            
            ['76_2_pre', 'FinalLiv-27_c002_v001_r001_reg001_he_aligned.tif', '076-2.ome.tif','cytassist_76_pre', 'FinalLiv-27_c002_v001_r001_reg001_codex.csv','076_2_tissue_positions_list.csv','FinalLiv-27_c002_v001_r001_reg001.cell_types.csv'],
            ['76_3_pre', 'FinalLiv-27_c002_v001_r001_reg002_he_aligned.tif', '076-3.ome.tif','cytassist_76_pre', 'FinalLiv-27_c002_v001_r001_reg002_codex.csv','076_3_tissue_positions_list.csv','FinalLiv-27_c002_v001_r001_reg002.cell_types.csv'],
            ['76_4_post','FinalLiv-27_c002_v001_r001_reg003_he_aligned.tif', '076-4.ome.tif','cytassist_76_post','FinalLiv-27_c002_v001_r001_reg003_codex.csv','076_4_tissue_positions_list.csv','FinalLiv-27_c002_v001_r001_reg003.cell_types.csv'],
            ['76_5_post','FinalLiv-27_c002_v001_r001_reg004_he_aligned.tif', '076-5.ome.tif','cytassist_76_post','FinalLiv-27_c002_v001_r001_reg004_codex.csv','076_5_tissue_positions_list.csv','FinalLiv-27_c002_v001_r001_reg004.cell_types.csv'],
            
            ['79_2_pre', 'FinalLiv-27_c001_v001_r001_reg029_he_aligned.tif', '079-2.ome.tif','cytassist_79_pre', 'FinalLiv-27_c001_v001_r001_reg029_codex.csv','079_2_tissue_positions_list.csv','FinalLiv-27_c001_v001_r001_reg029.cell_types.csv'],
            ['79_4_post','FinalLiv-27_c001_v001_r001_reg031_he_aligned.tif', '079-4.ome.tif','cytassist_79_post','FinalLiv-27_c001_v001_r001_reg031_codex.csv','079_4_tissue_positions_list.csv','FinalLiv-27_c001_v001_r001_reg031.cell_types.csv'],
            ['79_5_post','FinalLiv-27_c001_v001_r001_reg032_he_aligned.tif', '079-5.ome.tif','cytassist_79_post','FinalLiv-27_c001_v001_r001_reg032_codex.csv','079_5_tissue_positions_list.csv','FinalLiv-27_c001_v001_r001_reg032.cell_types.csv'],
            
            ['83_2_pre', 'FinalLiv-27_c002_v001_r001_reg005_he_aligned.tif', '083-2.ome.tif','cytassist_83_pre', 'FinalLiv-27_c002_v001_r001_reg005_codex.csv','083_2_tissue_positions_list.csv','FinalLiv-27_c002_v001_r001_reg005.cell_types.csv'],
            
            ['84_1_pre', 'FinalLiv-27_c002_v001_r001_reg008_he_aligned.tif', '084-1.ome.tif','cytassist_84_pre', 'FinalLiv-27_c002_v001_r001_reg008_codex.csv','084_1_tissue_positions_list.csv','FinalLiv-27_c002_v001_r001_reg008.cell_types.csv'],
            ['84_2_pre', 'FinalLiv-27_c002_v001_r001_reg009_he_aligned.tif', '084-2.ome.tif','cytassist_84_pre', 'FinalLiv-27_c002_v001_r001_reg009_codex.csv','084_2_tissue_positions_list.csv','FinalLiv-27_c002_v001_r001_reg009.cell_types.csv'],
            ['84_3_post','FinalLiv-27_c002_v001_r001_reg010_he_aligned.tif', '084-3.ome.tif','cytassist_84_post','FinalLiv-27_c002_v001_r001_reg010_codex.csv','084_3_tissue_positions_list.csv','FinalLiv-27_c002_v001_r001_reg010.cell_types.csv'],
            ['84_4_post','FinalLiv-27_c002_v001_r001_reg011_he_aligned.tif', '084-4.ome.tif','cytassist_84_post','FinalLiv-27_c002_v001_r001_reg011_codex.csv','084_4_tissue_positions_list.csv','FinalLiv-27_c002_v001_r001_reg011.cell_types.csv'],
                 
            ['85_3_pre', 'FinalLiv-27_c002_v001_r001_reg013_he_aligned.tif', '085-3.ome.tif','cytassist_85_pre', 'FinalLiv-27_c002_v001_r001_reg013_codex.csv','085_2_tissue_positions_list.csv','FinalLiv-27_c002_v001_r001_reg013.cell_types.csv']
           ]

protein_to_gene ={'ICOS':'ICOS',
                  'TOX':'TOX',
                  'CXCR5':'CXCR5',
                  'CD15':'FUT4',
                  'FAP':'FAP',
                  'CD56':'NCAM1',
                  'GRB':'GRB10',
                  'Podoplanin':'PDPN',
                  'IFNg':'IFNG',
                  'CD4':'CD4',
                  'CD107a':'LAMP1',
                  'PNAD':'NTAN1',
                  'TCF1':'TCF7',
                  'PD1':'PDCD1',
                  'Ki67':'MKI67',
                  #'KERATIN 8_18':'KRT18',
                  'CD31':'PECAM1',
                  'CD11c':'ITGAX',
                  'VISTA':'VSIR',
                  'CD8':'CD8A',
                  'FOXP3':'FOXP3',
                  #'HLADR':'HLADR',
                  'CD138':'SDC1', 
                'CD204':'MSR1', 
                'PDL1':'CD274', 
                'CD79a':'CD79A', 
                'Tbet':'TBX21', 
                'CCR7':'CCR7', 
                'CD141':'THBD', 
                'CD45RO':'PTPRC',
                'CD20':'MS4A1', 
                'eomes':'EOMES', 
                'IDO1':'IDO1', 
                'INOS':'NOS2', 
                'CD68':'CD68', 
                'DCLAMP':'LAMP3', 
                'CD163':'CD163', 
                'CD1c':'CD1C',
                'CD3':'CD3G', 
                'BCL6':'BCL6', 
                'XCR1':'XCR1', 
                'CXCL13':'CXCL13', 
                'CD209':'CD209', 
                'CD34':'CD34', 
                'CD14':'CD14', 
                'CD206':'MRC1',
                'aSMA':'ACTA2', 
                'LAG3':'LAG3', 
                'CD21':'CR2', 
                'CD45':'PTPRC'   
                 }

def return_protein_to_gene():
    return protein_to_gene


def load_data(visium_path, trans_loc, codex_path, type_path, cell_type_label_mapping, used_gene = 2000,select_gene=None):
    """
    no preprocessing, loading data only
    """
    visium_adata = sc.read_visium(visium_path)
    visium_adata.var_names_make_unique()
    transformed_loc = pd.read_csv(trans_loc,index_col=0)
    transformed_loc.index = transformed_loc['barcode']
    transformed_loc = transformed_loc.loc[visium_adata.obs_names,:]
    visium_adata.obs = transformed_loc
    #visium_adata = preprocess_visium(visium_adata, n_top_genes=used_gene)

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
    
    return visium_adata, codex_adata

def preprocess_codex_kmean(codex_adata, num_categories):
    """
    get kmean for codex data
    """
    codex_adata_copy = codex_adata.copy()
    sc.pp.log1p(codex_adata_copy)
    sc.pp.pca(codex_adata_copy)
    sc.pp.neighbors(codex_adata_copy)

    X_pca = codex_adata_copy.obsm['X_pca'] 

    kmeans = KMeans(n_clusters=num_categories, random_state=0).fit(X_pca) 
    codex_adata.obs['kmean'] = kmeans.labels_.astype(int)
    codex_adata.obsm = codex_adata_copy.obsm
    return codex_adata

def plot_spatial_visium(adata_df, codex_item,vmax=1,vmin=0):  
    plt.figure(figsize=(5,5),dpi=500)
    loc_info = adata_df.loc[:,['transformed coords x', 'transformed coords y']]
    fig, axs = plt.subplots(figsize = (5,5),dpi=500)
    axs.scatter(loc_info.iloc[:,0],-loc_info.iloc[:,1],
                s=55,
                c=adata_df['visium_'+protein_to_gene[codex_item]],
                vmax=vmax,
                vmin=vmin
               )
    plt.axis('off')

def plot_spatial_downsize_codex(adata_df, codex_item,vmax=1,vmin=0):  
    plt.figure(figsize=(5,5),dpi=500)
    loc_info = adata_df.loc[:,['transformed coords x', 'transformed coords y']]
    fig, axs = plt.subplots(figsize = (5,5),dpi=500)
    axs.scatter(loc_info.iloc[:,0],-loc_info.iloc[:,1],
                s=55,
                c=adata_df['codex_'+codex_item],
                vmax=vmax,
                vmin=vdisplay_reconstmin
               )
    plt.axis('off')
    
def plot_spatial_codex(codex_data, codex_item, vmax=0.5,vmin=0):    
    loc_info = codex_data.loc[:,['x', 'y']]
    fig, axs = plt.subplots(figsize = (5,5),dpi=500)
    axs.scatter(loc_info['x'],-loc_info['y'],s=0.3,
                c=codex_data[codex_item],
                vmax=vmax,
                vmin=vmin
               )
    plt.axis('off')
    
    
def load_codex_visium(visium_file,codex_file):
    return (sc.read_h5ad(visium_file), 
            pd.read_csv(codex_file,index_col=0)
           )

def preprocess_visium(visium_data,
                      used_gene=None,
                      select_gene= None,
                     ):
    """
    get the n top genes, and included codex related genes
    """
    
    visium_data.var_names_make_unique()
    
    ## delete MT genes and RPL/RPS gene
    visium_data = visium_data[:,[not ((i.startswith('MT-'))|(i.startswith('RPL'))|(i.startswith('RPS'))) for i in visium_data.to_df().columns]]
    
    adata = visium_data.copy()
    
    sc.pp.normalize_total(visium_data,target_sum=1e4,inplace=True)
    sc.pp.log1p(visium_data)
    
    sc.pp.highly_variable_genes(visium_data, n_top_genes=used_gene, inplace=True)
    
    PG = return_protein_to_gene()
    
    for i in PG.keys():
        if PG[i] in visium_data.var_names:
            visium_data.var.loc[PG[i], 'highly_variable'] = True


    if select_gene is not None:
        visium_data.var.loc[select_gene, 'highly_variable'] = True
    adata = adata[:, visium_data.var.highly_variable]
    adata.X = adata.X.toarray()
    
    return adata


# Apply quantile normalization
def quantile_normalize(df):
    rank_mean = df.stack().groupby(df.rank(method='first').stack().astype(int)).mean()
    #print(rank_mean)
    return df.rank(method='min').stack().astype(int).map(rank_mean).unstack()
 

def asinh_transform(x, cofactor=5):
    #return np.arcsinh(x / cofactor)
    return np.arcsinh(x / (cofactor * x.quantile(0.2)))


def preprocess_codex(codex_data_normed):
    #codex_data_normed  = quantile_normalize(codex_data.iloc[:,3:].transpose()).transpose()
    # Apply the asinh transformation to the protein expression data
    codex_data_normed = asinh_transform(codex_data_normed)
    codex_data_normed.fillna(0, inplace=True)
    codex_data_normed.replace([np.inf, -np.inf], 0, inplace=True)

    # Calculate the mean and standard deviation along each cell
    #mean_values = codex_data_normed.mean(axis=1)
    #std_values = codex_data_normed.std(axis=1)

    # Z-score normalize the data along axis 1
    #codex_data_normed = (codex_data_normed.transpose() - mean_values) / std_values
    #codex_data_normed = codex_data_normed.transpose()
    #codex_data_normed = codex_data_normed - codex_data_normed.min()
    #codex_data_normed[['cell_id','x','y']] = codex_data[['cell_id','x','y']]
    
    return codex_data_normed
    
def robust_z_scale(df, columns):
    
    scaled_df = df.copy()  
    for column in columns:
        median = np.median(df[column])
        q75, q25 = np.percentile(df[column], [90, 32])
        iqr = q75 - q25
        scaled_df[column] = (df[column] - median) / iqr
    
    return scaled_df

    
def shared_df(visium_data, codex_data):
    protein_to_gene = return_protein_to_gene()
    ## create new loc array for shared codex and visium 
    loc_xyab = visium_data.obs.loc[:,['transformed coords x', 'transformed coords y']]
    window_size = 80
    
    for codex_item in list(protein_to_gene.keys())[1:]:
        #codex_item = 'CD31'
        #print(codex_item)
        codex_spots_aligned_ = []
        for i in range(loc_xyab.shape[0]):
            x_ = loc_xyab.iloc[i,0]
            y_ = loc_xyab.iloc[i,1]
            codex_spots_aligned_.append(
                codex_data[codex_item][((codex_data['x']-x_)**2 + (codex_data['y']-y_)**2)<window_size**2].sum(axis=0)
            )
        loc_xyab['codex_'+codex_item] = codex_spots_aligned_
        loc_xyab['visium_'+protein_to_gene[codex_item]] = visium_data[:,protein_to_gene[codex_item]].to_df()
    return loc_xyab




## create new loc array for shared codex and visium
def find_similarity(loc_xyab,codex_item):
    #loc_array=np.array(loc_xyab)
    
    codex_array=loc_xyab['codex_'+codex_item]
    visium_array=loc_xyab['visium_'+protein_to_gene[codex_item]]
    cos_theta=np.sum(codex_array*visium_array)/(np.linalg.norm(codex_array)*np.linalg.norm(visium_array))
    return cos_theta

def get_simi_df(reg_vc):
    cos_simi = []
    for codex_item in list(return_protein_to_gene().keys())[1:]:
        cos_simi.append([codex_item,find_similarity(reg_vc,codex_item)])
    cos_simi= pd.DataFrame(cos_simi)
    cos_simi = cos_simi.sort_values(by=1,ascending=False)
    return cos_simi


def plot_compare(reg01_vc, reg01_codex_norm, codex_item,vmax1=1,vmax2=1,vmax3=1):
    utils.plot_spatial_visium(reg01_vc,
                          codex_item = codex_item,
                          vmax=vmax1
                         )
    utils.plot_spatial_downsize_codex(reg01_vc,
                          codex_item = codex_item,
                          vmax=vmax2
                         )
    
    utils.plot_spatial_codex(reg01_codex_norm,
                         codex_item = codex_item,
                         vmax=vmax3
                        )


def r2_score(x, y):
    # Dummy R2 score computation for the example
    return 1 - np.sum((x - y) ** 2) / np.sum((x - np.mean(x)) ** 2)


def scatter_with_density(x, y, idx):
    fig, ax = plt.subplots(1, 1, figsize=(3,3))
    # Calculate density estimation using Gaussian KDE
    kde = gaussian_kde(np.vstack([x, y]))
    density = kde(np.vstack([x, y]))
    # Map density values to colors using a rainbow colormap
    colors = cm.rainbow(density)
    # Create a scatter plot with colored points
    sc = ax.scatter(x, y, c=colors, s=20, cmap='rainbow')
    #sc = ax.scatter(x, y, s=20)
    ax.set_aspect('equal', adjustable='datalim')
    
    ax.set_xlabel("Actual")
    ax.set_ylabel("Reconstructed")

    # Calculate R-squared score
    r2 = scipy.stats.pearsonr(x, y).statistic
    
    # Add R-squared score as text on the plot
    ax.text(0.5, 0.9, f"Pearsonr: {r2:.2f}", transform=ax.transAxes,
            fontsize=12, color='black', ha='center', va='center')
    # Set axes to have the same limits
    combined_min = min(ax.get_xlim()[0], ax.get_ylim()[0])
    combined_max = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.set_xlim(combined_min, combined_max)
    ax.set_ylim(combined_min, combined_max)

    # Setting the aspect ratio to be equal, to ensure units are the same along both axes
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Codex Reconstruction for protein " + idx)
    plt.tight_layout()
    plt.show()
    
    
def display_reconst(df_true,
                    df_pred,
                    density=False,
                    marker_genes=None,
                    sample_rate=0.1,
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
    plt.figure(dpi=500)
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
    
    
    
def get_cmp(num_categories):
    rainbow_cmap = plt.get_cmap('rainbow')
    # Determine the number of unique categories in your data

    # Sample colors from the rainbow colormap
    colors = [rainbow_cmap(i / num_categories) for i in range(num_categories)]

    # Shuffle the order of the colors
    np.random.shuffle(colors)
    #colors = [rainbow_cmap(i/len(np.unique(np.argmax(all_ci_values, axis=1)))) for i in range(len(np.unique(np.argmax(all_ci_values, axis=1))))]
    #colors=['red', 'green', 'blue', 'purple', 'orange', 'pink', 'cyan', 'magenta', 'yellow', 'lime'] # Add more colors as needed
    #colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    cmap = ListedColormap(colors)
    return cmap


def manual_colors():
    colors = [
        '#0575f5',#8
        '#40bfe3',#9
        '#17bf2a',#10
        '#83eb93', #18
        '#f5700a',
        '#f5962a',
        '#eb0202',
        '#f58e8e',
        '#d923a2',
        '#f743c1',
        '#8c4637',
        '#997673',
        '#5c5a5a',
        '#8f8d8d'
]
    return colors
    
def get_predicted_data(coral_model):   
    sample_ids, visium_paths, trans_locs, codex_paths, type_paths = utils_exp.sample_lists()

    type_all_sets = set()
    for type_path in type_paths:
        temp = pd.read_csv(type_path,index_col=0)
        type_all_sets = type_all_sets | set(temp['CELL_TYPE'].unique())
        #print(temp['CELL_TYPE'].unique())
    cell_type_label_mapping = {}
    for j,i in enumerate(type_all_sets):
        cell_type_label_mapping[i]=int(j)
    adata_pred_sc_visium = anndata.AnnData(np.array(coral_model.eval_px))
    adata_pred_codex = anndata.AnnData(np.array(coral_model.eval_py))
    
    predicted_states = coral_model.all_vi_values
    predicted_states = np.array(predicted_states)

    annotated_types = torch.Tensor(list(coral_model.all_codex_type))
    annotated_types = np.array(annotated_types).astype(int)

    reversed_mapping = {v: k for k, v in cell_type_label_mapping.items()}

    adata_pred_sc_visium.obsm['qv'] = predicted_states
    adata_pred_sc_visium.obs['cell_type'] = annotated_types
    adata_pred_sc_visium.obs['cell_type'] = adata_pred_sc_visium.obs['cell_type'].astype('category').map(reversed_mapping)

    adata_pred_sc_visium.obs['sample'] = coral_model.all_sample_id
    adata_pred_sc_visium.obs['sample'] = adata_pred_sc_visium.obs['sample'].astype(int).astype('category')

    if coral_model.gene_list is not None:
        adata_pred_sc_visium.var_names = coral_model.gene_list
    if coral_model.protein_list is not None:
        adata_pred_codex.var_names = coral_model.protein_list
    
    adata_pred_sc_visium.obs[['x','y']] = np.array(coral_model.all_cell_xy)
    
    adata_pred_sc_visium.obsm['qz'] = coral_model.all_zi_values
    
    flatten_spot_id = []
    for i in range(len(coral_model.spot_id_cell)):
        flatten_spot_id.append(torch.concat(coral_model.spot_id_cell[i]))
    flatten_spot_id = torch.cat(flatten_spot_id)
    adata_pred_sc_visium.obs['spot_id'] = np.array(flatten_spot_id)
    
    return adata_pred_sc_visium, adata_pred_codex