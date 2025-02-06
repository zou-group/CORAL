import torch
from torch import nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from . import utils 
import matplotlib.pyplot as plt
from torch.distributions import Distribution, constraints, Normal,LogNormal, Gamma, Poisson
from torch import Tensor
from torch_geometric.typing import Adj, OptPairTensor, Size
from typing import Optional, Tuple

    
class CrossAttentionLayer(nn.Module):
    def __init__(self, latent_dim):
        super(CrossAttentionLayer, self).__init__()
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.key_proj = nn.Linear(latent_dim, latent_dim)
        self.value_proj = nn.Linear(latent_dim, latent_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        queries = self.query_proj(query)
        keys = self.key_proj(key)
        values = self.value_proj(value)
        attention_weights = self.softmax(torch.bmm(queries, keys.transpose(1, 2)))
        attended_output = torch.bmm(attention_weights, values)
        return attended_output

    
class DeconvolutionLayer(nn.Module):
    def __init__(self, input_dim, cell_type_dim, output_dim):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim+cell_type_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        
        return self.fc(x)
    
class CORAL_model(nn.Module):
    def __init__(self, visium_dim, codex_dim, cell_type_dim, latent_dim, hidden_channels, v_dim,  eps=1e-10):

        super().__init__()
        
        self.visium_dim = visium_dim
        self.codex_dim = codex_dim
        self.cell_type_dim = cell_type_dim
        self.latent_dim = latent_dim
         
        self.encoder_visium = nn.Sequential(
            nn.Linear(visium_dim+cell_type_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim *2)  # Outputting both mu and logvar for z
        )

        self.encoder_codex = nn.Sequential(
            nn.Linear(codex_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim *2)  # Outputting both mu and logvar for z
        )
        
        self.encoder = nn.Sequential(
            nn.Linear(visium_dim + codex_dim + cell_type_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)
        )
        
        self.cross_attention = CrossAttentionLayer(latent_dim)

        
        self.deconv = DeconvolutionLayer(visium_dim, cell_type_dim, visium_dim)
        self.gat1 = GATConv(latent_dim, hidden_channels, heads=4, concat=True, dropout=0.5)
        self.gat2 = GATConv(hidden_channels * 4, latent_dim, heads=1, concat=False, dropout=0.5)
        
        
        self.hidden_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU()
        )
        
        self.visium_scale_decoder = nn.Sequential(
                                nn.Linear(128, visium_dim),
                                nn.Softmax(dim=-1)
        )
        
        self.codex_scale_decoder = nn.Sequential(
                                nn.Linear(128, codex_dim),
                                nn.Softmax(dim=-1)
        )
        self.cell_type_decoder = nn.Linear(latent_dim, cell_type_dim)
        
        self.v_m_layer = nn.Linear(latent_dim + cell_type_dim, v_dim)  
        self.v_logvar_layer = nn.Linear(latent_dim + cell_type_dim,  v_dim)  
        
        self.eps = eps
        
        # Distribution parameters
        self._px_r = torch.nn.Parameter(torch.randn(visium_dim), requires_grad=True)
        self._px_r_sc = torch.nn.Parameter(torch.randn(visium_dim), requires_grad=True)
        self._py_r = torch.nn.Parameter(torch.randn(codex_dim), requires_grad=True)
        
        
           
    @property
    def px_r(self):
        return F.softplus(self._px_r) + self.eps
    @property
    def px_r_sc(self):
        return F.softplus(self._px_r_sc) + self.eps
        
    @property
    def py_r(self):
        return F.softplus(self._py_r) + self.eps


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    #def encode(self, x):
    #    h = self.encoder(x)
    #    mu, logvar = h.chunk(2, dim=-1)
    #    z = self.reparameterize(mu, logvar)
    #    return z, mu, logvar
    
    def encode(self, x_combined, cell_type, edge_index):
        
        
        visium_h = self.encoder_visium(torch.cat((x_combined[:, :self.visium_dim], cell_type), dim=1))
        codex_h = self.encoder_codex(x_combined[:, self.visium_dim:])
        visium_mu, visium_logvar = visium_h.chunk(2, dim=-1)
        codex_mu, codex_logvar = codex_h.chunk(2, dim=-1)
        
        #x = torch.cat((x_combined, cell_type), dim=1)
        #h = self.encoder(x)
        #mu, logvar = h.chunk(2, dim=-1)
        #z = self.reparameterize(mu, logvar)
        
        #return z, mu, logvar
        
        
        visium_mu, _ = self.gat1(visium_mu, edge_index, return_attention_weights=True)
        visium_mu = torch.relu(visium_mu)
        visium_mu, _ = self.gat2(visium_mu, edge_index, return_attention_weights=True)

        codex_mu, _ = self.gat1(codex_mu, edge_index, return_attention_weights=True)
        codex_mu = torch.relu(codex_mu)
        codex_mu, _ = self.gat2(codex_mu, edge_index, return_attention_weights=True)

        attended_codex_mu = self.cross_attention(codex_mu.unsqueeze(1), visium_mu.unsqueeze(1), visium_mu.unsqueeze(1)).squeeze(1)
        attended_visium_mu = self.cross_attention(visium_mu.unsqueeze(1), codex_mu.unsqueeze(1), codex_mu.unsqueeze(1)).squeeze(1)
        
        attended_codex_logvar = self.cross_attention(codex_logvar.unsqueeze(1), visium_logvar.unsqueeze(1), visium_logvar.unsqueeze(1)).squeeze(1)
        attended_visium_logvar = self.cross_attention(visium_logvar.unsqueeze(1), codex_logvar.unsqueeze(1), codex_logvar.unsqueeze(1)).squeeze(1)
        
        
        combined_mu = (attended_codex_mu + attended_visium_mu) / 2
        
        combined_logvar = (attended_codex_logvar + attended_visium_logvar) / 2
        combined_latent = self.reparameterize(combined_mu, combined_logvar)
        return combined_latent, combined_mu, combined_logvar
    
    def infer_v(self, z, cell_type):
        z_with_type = torch.cat((z, cell_type), dim=1)  # Concatenate z with cell type
        v_m = self.v_m_layer(z_with_type)
        v_logvar = self.v_logvar_layer(z_with_type)
        v = self.reparameterize(v_m, v_logvar)
        return v, v_m, v_logvar


    def forward(self, batch):
        """
        batch: subgraph for each cell
        """
        
        x, edge_index, cell_type, spot_indices,visium_spot = batch.x, batch.edge_index, batch.cell_type, batch.spot_indices, batch.visium_spot
        
        visium_x, codex_x = x[:, :self.visium_dim], x[:, self.visium_dim:]
        codex_library  = torch.log1p(codex_x.sum(dim=1)).unsqueeze(1)
        visium_library = torch.log1p(visium_x.sum(dim=1)).unsqueeze(1)
        
        #q_xi_m
        #visium_x_with_type = torch.cat((visium_x, cell_type), dim=1)
        deconv_x = self.deconv(torch.cat((torch.log1p(visium_x), cell_type), dim=1)) 
        #deconv_x = self.deconv(torch.log1p(visium_x_with_type))
        
        num_ = [(spot_indices == spot_).sum() for spot_ in visium_spot]
        mean_num = torch.tensor(num_,dtype=torch.float64).mean()
        
        
        q_xi_dist = NegBinom(torch.exp(visium_library/mean_num)*deconv_x+ self.eps,torch.exp(self.px_r_sc))
        q_xi = q_xi_dist.sample()
        

        x_combined = torch.cat((q_xi, codex_x), dim=1)
        x_combined = torch.log1p(x_combined)
        
        z, z_mu, z_logvar = self.encode(x_combined, cell_type, edge_index)
        
        v, v_m, v_logvar = self.infer_v(x_combined, z, cell_type)

        
        v, v_m, v_logvar = self.infer_v(z, cell_type)
        
        # Apply GAT layers
        z, attn_weights_1 = self.gat1(z, edge_index, return_attention_weights=True)
        z = torch.relu(z) 
        z, attn_weights_2 = self.gat2(z, edge_index, return_attention_weights=True)
        
        
        hidden = self.hidden_decoder(z)
        # Decode for visium part using negative binomial distribution
        visium_scale = self.visium_scale_decoder(hidden)
        visium_rate = visium_scale * torch.exp(visium_library)  + self.eps
        
        # Decode for codex part using gamma distribution
        codex_scale = self.codex_scale_decoder(hidden)
        codex_rate = codex_scale * torch.exp(codex_library) + self.eps

        # Decode for cell type
        generated_cell_type = self.cell_type_decoder(z)
        # Aggregate single-cell visium data to spot level
        aggregated_visium = torch.zeros((len(visium_spot), visium_rate.shape[1]), device=visium_rate.device)
        for j, spot_ in enumerate(visium_spot):
            # Sum the visium_rate for all cells that belong to the current spot
            aggregated_visium[j] = visium_rate[spot_indices == spot_].mean(dim=0)
            
        return {
            
            'pxi_rate':visium_rate,
            'px_rate_aggregated': aggregated_visium,
            'px_r': self.px_r,
            
            'py_rate': codex_rate,
            'py_r': self.py_r,
            
            'z_mu': z_mu,
            'z_logvar': z_logvar,
            'z': z,  
            
            'generated_cell_type': generated_cell_type,
            
            'px_r_sc':self.px_r_sc,
            'q_xi_m':deconv_x,
            'q_xi':q_xi,
            
            'v': v,
            'v_m': v_m,
            'v_logvar': v_logvar,
            
            
            'edge_index': edge_index,  # Return edge_index for laplacian regularization
            'attn_weights_1': attn_weights_1,  # Attention weights from the first GAT layer
            'attn_weights_2': attn_weights_2   # Attention weights from the second GAT layer

            
        }
    
    
    def efficient_contrastive_loss(self, outputs, labels, margin=50):
        pairwise_dist = torch.pdist(outputs, p=2)
        triu_indices = torch.triu_indices(labels.size(0), labels.size(0), 1)
        labels_a = labels[triu_indices[0]]
        labels_b = labels[triu_indices[1]]
        is_positive = labels_a == labels_b
        is_negative = labels_a != labels_b
        positive_pairs = pairwise_dist[is_positive]
        negative_pairs = pairwise_dist[is_negative]
        positive_loss = positive_pairs.sum()
        negative_loss = F.relu(margin - negative_pairs).sum()
        n_pos = positive_pairs.size(0)
        n_neg = negative_pairs.size(0)
        loss = (positive_loss / n_pos) + (negative_loss / n_neg) if n_pos > 0 and n_neg > 0 else torch.tensor(0.0)
        return loss
    
    
    
    
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
    
    