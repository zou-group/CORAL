import torch
import torch.optim as optim
import torch.nn as nn
from .model import CORAL_model
from typing import Dict, Tuple, List
import logging
import numpy as np
from torch.distributions import Distribution,constraints,  Normal, LogNormal,Gamma, Poisson, Categorical, kl_divergence as kl
import random
from torch_geometric.utils import to_dense_adj

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

def create_model(visium_dim, codex_dim, cell_type_dim, latent_dim=50, hidden_channels=16, v_dim=10):
    model = CORAL_model(visium_dim, codex_dim, cell_type_dim, latent_dim, hidden_channels, v_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer

class NegBinom(Distribution):
    """
    Gamma-Poisson mixture approximation of Negative Binomial(mean, dispersion)

    lambda ~ Gamma(mu, theta)
    x ~ Poisson(lambda)
    """
    arg_constraints = {
        'mu': constraints.greater_than_eq(0),
        'theta': constraints.greater_than_eq(0),
    }
    support = constraints.nonnegative_integer

    def __init__(self, mu, theta, device='cuda', eps=1e-5):
        """
        Parameters
        ----------
        mu : torch.Tensor
            mean of NegBinom. distribution
            shape - [# genes,]

        theta : torch.Tensor
            dispersion of NegBinom. distribution
            shape - [# genes,]
        """
        device = device
        self.mu = mu.to(device)
        self.theta = theta.to(device)
        self.eps = eps
        self.device = device
        super(NegBinom, self).__init__(validate_args=True)

    def sample(self,sample_shape=torch.Size()):
        lambdas = Gamma(
            concentration=self.theta + self.eps,
            rate=(self.theta + self.eps) / (self.mu + self.eps),
        ).rsample(sample_shape).to(self.device)

        x = Poisson(lambdas).sample()

        return x

    def log_prob(self, x):
        x = x.to(self.device)
        """log-likelihood"""
        ll = torch.lgamma(x + self.theta) - \
             torch.lgamma(x + 1) - \
             torch.lgamma(self.theta) + \
             self.theta * (torch.log(self.theta + self.eps) - torch.log(self.theta + self.mu + self.eps)) + \
             x * (torch.log(self.mu + self.eps) - torch.log(self.theta + self.mu + self.eps))

        return ll
    
    
def graph_laplacian_regularization(edge_index, embeddings):
    # Get the edge weights if not already included in the edge_index
    edge_weight = torch.ones(edge_index.size(1), device=embeddings.device)
    
    # Compute the Laplacian (without converting to dense)
    row, col = edge_index
    degree = torch.bincount(row, minlength=embeddings.size(0)).float()
    degree_matrix = torch.diag(degree)
    
    # Laplacian matrix: L = D - A
    laplacian_matrix = degree_matrix - to_dense_adj(edge_index, edge_attr=edge_weight).squeeze(0)
    
    # Regularization term: 0.5 * Tr(embeddings.T @ L @ embeddings)
    laplacian_loss = 0.5 * torch.trace(embeddings.T @ laplacian_matrix @ embeddings)
    
    return laplacian_loss
    
def loss_function(model, visium_true, output, codex_true, cell_type_true, cell_type_pred, mu, logvar, contrastive_outputs, edge_weights,margin=50):


    epsilon = 1e-5

    # KLD for z
    q_z = Normal(output['z_mu'], torch.exp(0.5 * output['z_logvar']))
    p_z = Normal(output['p_zi_m'], torch.exp(0.5 * output['p_zi_logvar']))
    KLD_z = kl(q_z, p_z).sum(dim=1).mean()
    
    
    # KLD for v
    q_v = Normal(output['v_m'], torch.exp(0.5 * output['v_logvar']))
    p_v = Normal(torch.zeros_like(output['v_m']), torch.ones_like(output['v_logvar'])/2)
    KLD_v = kl(q_v, p_v).sum(dim=1).mean()

    #print(KLD)
    # Cross-entropy loss for cell type
    #cell_type_loss = nn.CrossEntropyLoss()(cell_type_pred, torch.argmax(cell_type_true, dim=1))
    
    # Negative binomial loss for Visium part
    visium_recon_loss = -NegBinom(output['px_rate_aggregated'], torch.exp(output['px_r'])).log_prob(visium_true).sum(-1).mean() 
    
    contrastive_loss = model.efficient_contrastive_loss(contrastive_outputs, torch.argmax(cell_type_true, dim=1), margin)
    
    
    xi_recon_loss  = - NegBinom(output['pxi_rate'], torch.exp(output['px_r_sc'])).log_prob(output['q_xi']).sum(-1).mean() 
    
    # Gamma loss for CODEX part
    codex_recon_loss = -Gamma(output['py_rate'], torch.exp(output['py_r'])).log_prob(codex_true+epsilon).sum(-1).mean()
    #print(output['py_rate'])
    
    laplacian_reg = graph_laplacian_regularization(output['edge_index'], output['z'])
    #laplacian_reg = graph_laplacian_regularization(output['edge_index'], output['z'], edge_weight=edge_weights)
    
    # Sum all losses
    #total_loss = 1e3 * visium_recon_loss + codex_recon_loss  + KLD_z + 0.1 * KLD_v + 1e2 * contrastive_loss + xi_recon_loss + 1e2*laplacian_reg
    total_loss = 1e3 * visium_recon_loss + codex_recon_loss  + KLD_z + 0.1 * KLD_v + 1e4 * contrastive_loss + xi_recon_loss + 1e2*laplacian_reg #heterogenous samples

    return total_loss

def spatial_attention(edge_index, spatial_coords):
    """
    Compute edge weights based on spatial proximity.
    
    Parameters:
    - edge_index: Tensor of shape (2, num_edges), representing the indices of the connected nodes.
    - spatial_coords: Tensor of shape (num_nodes, 2), representing the x, y coordinates of each node.
    
    Returns:
    - edge_weights: Tensor of shape (num_edges,), representing the computed edge weights.
    """
    num_edges = edge_index.size(1)
    edge_weights = torch.zeros(num_edges).to(spatial_coords.device)
    
    for i in range(num_edges):
        src, dst = edge_index[:, i]
        spatial_dist = torch.norm(spatial_coords[src] - spatial_coords[dst])
        edge_weights[i] = torch.exp(-spatial_dist)  # Exponential decay with distance
    
    return edge_weights

def train_model(model, optimizer, dataloader, epochs=300,device='cuda'):
    
    model.to(device)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    for epoch in range(epochs):
        model.train()
        
        train_loss = 0
        
        for step, batch in enumerate(dataloader):
            
            batch = batch.to(device)
            
            edge_index = batch.edge_index
            spatial_coords = batch.spatial_coords
            center_cell = batch.center_cell
            cell_type = batch.cell_type
            visium_true, codex_true = batch.visium_spot_exp, batch.x[:, model.visium_dim:]
             # Compute edge weights for this specific batch
            #edge_weights = spatial_attention(edge_index, spatial_coords)
            
            output = model(batch)
            
            codex_true = codex_true[center_cell]
            cell_type = cell_type[center_cell]
            
            loss = loss_function(model, visium_true, output, codex_true, cell_type, output['generated_cell_type'], output['z_mu'], output['z_logvar'],output['pxi_rate'], output['attn_weights_2'])
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            train_loss += loss.item()
            
        

        if (epoch + 1) % 1 == 0:
            print(f'Epoch {epoch}, Loss: {train_loss / len(dataloader)}')
        
        
        scheduler.step()
        
def generate_and_validate(model, dataloader,device):
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

        
            output = model(batch)

            
            center_cell_idx = batch.center_cell.nonzero(as_tuple=True)[0]
            

            
            generated_expr.append(output['px_rate_aggregated'].cpu().numpy())
            generated_protein.append(output['py_rate'].cpu().numpy())
            latent_rep.append(output['z_mu'].cpu().numpy())
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
    visium_true = np.concatenate(visium_true, axis=0)
    codex_true = np.concatenate(codex_true, axis=0)
    v_values = np.concatenate(v_values, axis=0)
    cell_types = np.concatenate(cell_types, axis=0)
    
    edges_all = np.concatenate(edge_indices_all, axis=1)
    attn_weights_all = np.concatenate(attn_weights_all, axis=0)
    
    
    return generated_expr, generated_protein, latent_rep, locations, visium_true,codex_true,attn_weights_all, edges_all, v_values, cell_types 


def main(adata_downsampled, adata_adt):
    # Step 1: Data Preprocessing
    combined_expr, visium_coords = utils.preprocess_data(adata_smoothed, adata_adt)
    
    # Step 2: Define Model
    input_dim = combined_expr.shape[1]
    model, optimizer = utils.create_model(input_dim)
    
    # Step 3: Prepare Data
    dataloader = utils.prepare_data(combined_expr, codex_coords)
    
    # Step 4: Train Model
    train_model(model, optimizer, dataloader)
    
    # Step 5: Generate and Validate Data
    generated_expr, latent_rep = generate_and_validate(model, dataloader)
    
    return generated_expr, latent_rep, model 

#generated_expr, latent_rep, model = main(adata_smoothed, adata_adt)
