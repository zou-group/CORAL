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
    adj = to_dense_adj(edge_index).squeeze(0)
    d = torch.diag(adj.sum(1))
    laplacian = d - adj
    return 0.5 * torch.trace(embeddings.T @ laplacian @ embeddings)
    
    
    
def loss_function(model, visium_true, output, codex_true, cell_type_true, cell_type_pred, mu, logvar, contrastive_outputs, contrastive_labels, margin=50):


    epsilon = 1e-5

    # KLD for z
    q_z = Normal(output['z_mu'], torch.exp(0.5 * output['z_logvar']))
    p_z = Normal(output['v'], torch.ones_like(output['v']))
    KLD_z = kl(q_z, p_z).sum(dim=1).mean()
    
    
    # KLD for v
    q_v = Normal(output['v_m'], torch.exp(0.5 * output['v_logvar']))
    p_v = Normal(torch.zeros_like(output['v_m']), torch.exp(torch.ones_like(output['v_logvar'])/2))
    KLD_v = kl(q_v, p_v).sum(dim=1).mean()

    #print(KLD)
    # Cross-entropy loss for cell type
    cell_type_loss = nn.CrossEntropyLoss()(cell_type_pred, torch.argmax(cell_type_true, dim=1))
    
    # Negative binomial loss for Visium part
    visium_recon_loss = -NegBinom(output['px_rate_aggregated'], torch.exp(output['px_r'])).log_prob(visium_true).sum(-1).mean() 
    
    contrastive_loss = model.efficient_contrastive_loss(contrastive_outputs, torch.argmax(contrastive_labels, dim=1), margin)
    
    
    xi_recon_loss  = - NegBinom(output['pxi_rate'], torch.exp(output['px_r_sc'])).log_prob(output['q_xi']).sum(-1).mean() 
    
    # Gamma loss for CODEX part
    codex_recon_loss = -Gamma(output['py_rate'], torch.exp(output['py_r'])).log_prob(codex_true+epsilon).sum(-1).mean()
    #print(output['py_rate'])
    
    laplacian_reg = graph_laplacian_regularization(output['edge_index'], output['z'])

    
    
    # Sum all losses
    total_loss = 1e3 * visium_recon_loss + codex_recon_loss  + KLD_z + KLD_v + contrastive_loss + xi_recon_loss + laplacian_reg

    return total_loss

def train_model(model, optimizer, dataloader, epochs=300,device='cuda'):
    
    
    model.to(device)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    for epoch in range(epochs):
        model.train()
        
        train_loss = 0
        
        for step, batch in enumerate(dataloader):
            
            batch = batch.to(device)
            output = model(batch)
            
            
            visium_true, codex_true = batch.visium_spot_exp, batch.x[:, model.visium_dim:]
            
            
            
            contrastive_outputs = output['pxi_rate']  # or use another output from the model as needed
            contrastive_labels = batch.cell_type  # or use another label as needed
            
            loss = loss_function(model, visium_true, output, codex_true, batch.cell_type, output['generated_cell_type'], output['z_mu'], output['z_logvar'],contrastive_outputs, contrastive_labels)
            

            
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
    
    v_values = []
    cell_types = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch)
            
            center_cell_idx = batch.center_cell.nonzero(as_tuple=True)[0]
            
            center_latent_rep = output['z_mu'][center_cell_idx]
            center_generated_protein_ = output['py_rate']#[center_cell_idx]
            
            visium_true_, codex_true_ = batch.visium_spot_exp, batch.x[:, model.visium_dim:]
            codex_true_ = codex_true_#[center_cell_idx]
            
            
            generated_expr.append(output['px_rate_aggregated'].cpu().numpy())
            generated_protein.append(center_generated_protein_.cpu().numpy())
            latent_rep.append(center_latent_rep.cpu().numpy())
            visium_true.append(visium_true_.cpu().numpy())
            codex_true.append(codex_true_.cpu().numpy())
            attn_weights_all.append((output['attn_weights_1'][1].cpu().numpy(), output['attn_weights_2'][1].cpu().numpy()))
            
            v_values.append(output['v'][center_cell_idx].cpu().numpy())
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
    
    return generated_expr, generated_protein, latent_rep, locations, visium_true,codex_true,attn_weights_all, v_values, cell_types


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
