import torch
import torch.optim as optim
import torch.nn as nn
from typing import Dict, Tuple, List
import logging
import numpy as np
from torch.distributions import Distribution,constraints,  Normal, LogNormal,Gamma, Poisson, Categorical, kl_divergence as kl
import random
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F
    
def protein_rna_contrastive_loss(z_protein: torch.Tensor,
                                 z_rna: torch.Tensor,
                                 temperature: float = 0.1) -> torch.Tensor:
    """
    Symmetric InfoNCE contrastive loss between protein and RNA embeddings.

    z_protein: (B, D)
    z_rna:     (B, D)
    """
    # Normalize
    z_protein = F.normalize(z_protein, dim=-1)
    z_rna = F.normalize(z_rna, dim=-1)

    # Similarity matrix (B, B)
    logits = torch.matmul(z_protein, z_rna.t()) / temperature

    batch_size = z_protein.size(0)
    labels = torch.arange(batch_size, device=z_protein.device)

    loss_i = F.cross_entropy(logits, labels)      # protein -> RNA
    loss_j = F.cross_entropy(logits.t(), labels)  # RNA -> protein

    loss = 0.5 * (loss_i + loss_j)
    return loss
    
def diversity_loss(embeddings, spot_indices):
    unique_spots = torch.unique(spot_indices)
    loss = 0.0
    valid_spot_count = 0
    for spot in unique_spots:
        spot_embeds = embeddings[spot_indices == spot]
        if spot_embeds.size(0) > 1:
            loss -= torch.pdist(spot_embeds).mean()
            valid_spot_count += 1
    # Avoid division by zero
    if valid_spot_count > 0:
        loss = loss / valid_spot_count
    else:
        loss = torch.tensor(0.0, device=embeddings.device)
    return loss




def graph_laplacian_regularization(edge_index, embeddings):
    row, col = edge_index
    diff = embeddings[row] - embeddings[col]  # (E, embedding_dim)
    laplacian_loss = (diff ** 2).sum() / edge_index.size(1)
    return laplacian_loss


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

    def __init__(self, mu, theta, device='cpu', eps=1e-5):
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
        #print(device)
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
    

def train_model(model, optimizer, dataloader, epochs=300,device='cpu', prot_idx = None, gene_idx = None, granu = True, heterogeneity = False):
    
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
            
            output = model(batch, device)
            
            codex_true = codex_true[center_cell]
            cell_type = cell_type[center_cell]
            
            loss = loss_function(model, batch.spot_indices, visium_true, output, codex_true, cell_type, output['generated_cell_type'], output['z_mu'], output['z_logvar'],output['pxi_rate'], output['attn_weights_2'], granu = granu, heterogeneity = heterogeneity)


            # ---- 2) Contrastive loss between matched protein/RNA markers ----
            if prot_idx is not None and gene_idx is not None:
                # assume batch has x_protein, x_rna as tensors
                x_prot = output['py_rate'][:, prot_idx]   # (B, K)
                x_rna  = output["q_xi"][:, gene_idx]       # (B, K')

                # Only compute if shapes are valid & both modalities present
                if x_prot.numel() > 0 and x_rna.numel() > 0:
                    
                    c_loss = protein_rna_contrastive_loss(
                        x_prot, x_rna, temperature=0.1
                    )
                    loss = loss + 0.1 * c_loss      

            
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

    
            train_loss += loss.item()
            
        

        if (epoch + 1) % 1 == 0:
            print(f'Epoch {epoch}, Loss: {train_loss / len(dataloader)}')
        
        
        scheduler.step()



def loss_function(model, spot_indices, visium_true, output, codex_true, cell_type_true, cell_type_pred, mu, logvar, contrastive_outputs, edge_weights,margin=50, granu = True, heterogeneity=False):


    epsilon = 1e-5

    # KLD for z
    q_z = Normal(output['z_mu']+epsilon, torch.exp(0.5 * output['z_logvar']))
    p_z = Normal(output['p_zi_m']+ epsilon, torch.exp(0.5 * output['p_zi_logvar']))
    KLD_z = kl(q_z, p_z).sum(dim=1).mean()
    
    
    # KLD for v
    q_v = Normal(output['v_m']+ epsilon, torch.exp(0.5 * output['v_logvar']))
    p_v = Normal(torch.zeros_like(output['v_m']), torch.ones_like(output['v_logvar'])/2)
    KLD_v = kl(q_v, p_v).sum(dim=1).mean()

    #print(KLD)
    # Cross-entropy loss for cell type
    #cell_type_loss = nn.CrossEntropyLoss()(cell_type_pred, torch.argmax(cell_type_true, dim=1))
    
    # Negative binomial loss for low-res data part
    if model.low_res_data_dist=='NB':
        visium_recon_loss = -NegBinom(output['px_rate_aggregated'], torch.exp(output['px_r'])).log_prob(visium_true).sum(-1).mean() 
        xi_recon_loss  = - NegBinom(output['pxi_rate'], torch.exp(output['px_r_sc'])).log_prob(output['q_xi']).sum(-1).mean() 
        
    elif model.low_res_data_dist=='Poisson':
        visium_recon_loss = -Poisson(
        output['px_rate_aggregated']
    ).log_prob(visium_true).sum(-1).mean()
        xi_recon_loss  = - Poisson(
            output['pxi_rate']).log_prob(output['q_xi']).sum(-1).mean() 
    else:
        raise ValueError(
            f"Unsupported distribution '{model.low_res_data_dist}'. "
            "Choose 'NB' or 'Poisson'."
        )

    # Gamma loss for CODEX part
    if model.high_res_data_dist=='Gamma':
        codex_recon_loss = -Gamma(output['py_rate'], torch.exp(output['py_r'])).log_prob(codex_true+epsilon).sum(-1).mean()
    elif model.high_res_data_dist=='NB':
        codex_recon_loss = -NegBinom(output['py_rate'], torch.exp(output['py_r'])).log_prob(codex_true+epsilon).sum(-1).mean() 
    else:
        raise ValueError(
            f"Unsupported distribution '{model.high_res_data_dist}'. "
            "Choose 'NB' or 'Gamma'."
        )
        
    contrastive_loss = model.efficient_contrastive_loss(contrastive_outputs, torch.argmax(cell_type_true, dim=1), margin)
    
    
    
    
    laplacian_reg = graph_laplacian_regularization(output['edge_index'], output['z'])
    
    
    div_loss = diversity_loss(output['z'], spot_indices)
    #print(div_loss)
    # Sum all losses
    #if model.codex_size/model.visium_size>4:     
    if heterogeneity:
        total_loss = 1e3 * visium_recon_loss + codex_recon_loss  + KLD_z + 0.1 * KLD_v + 1e4 * contrastive_loss + xi_recon_loss + 1e2*laplacian_reg 
    elif granu:
        total_loss = 1e3 * visium_recon_loss + codex_recon_loss  + KLD_z + 0.1 * KLD_v + 1e2 * contrastive_loss + xi_recon_loss + 1e2*laplacian_reg
    else:
        total_loss = 1e3 * visium_recon_loss + codex_recon_loss  + KLD_z + 0.1 * KLD_v + 1 * contrastive_loss + xi_recon_loss + 1e3*laplacian_reg
    #else:
    #total_loss = 1e3 * visium_recon_loss + codex_recon_loss  + KLD_z + 0.1 * KLD_v + 1e4 * contrastive_loss + xi_recon_loss + 1e2*laplacian_reg #+ 1e2 * div_loss#heterogenous samples
    return total_loss