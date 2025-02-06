import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Dirichlet, constraints,  Normal,LogNormal, Gamma, Poisson, NegativeBinomial, Categorical,kl_divergence as kl
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
from torch_geometric.data import Batch
import random
import numpy as np
from sklearn.decomposition import PCA

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    

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

    def __init__(self, mu, theta, device, eps=1e-5):
        """
        Parameters
        ----------
        mu : torch.Tensor
            Mean of Negative Binomial distribution (shape: [# genes,]).
        theta : torch.Tensor
            Dispersion of Negative Binomial distribution (shape: [# genes,]).
        device : torch.device
            Device to run the operations on.
        eps : float, optional
            Small value to avoid division by zero.
        """
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

        
class Zprior(nn.Module):
    """
    z ~ p(z|c,u)
    """
    def __init__(self, k_dim, u_dim, hidden_dim, z_dim, v_dim):
        super(Zprior, self).__init__()

        self.network =nn.Sequential( 
            nn.Linear(v_dim+k_dim, hidden_dim ),
            nn.BatchNorm1d(hidden_dim, momentum=0.1, eps=1e-5),
            nn.ReLU()                        
            )
        self.mu_layer = nn.Linear(hidden_dim, z_dim)
        self.log_var_layer = nn.Linear(hidden_dim, z_dim)

    def forward(self, v,c):

        hidden = self.network(torch.cat([v, c], dim=1))
        mu = self.mu_layer(hidden)
        log_var = self.log_var_layer(hidden)
        return mu, log_var

class Xiprior(nn.Module):
    """
    z ~ p(z|c,u)
    """
    def __init__(self, z_dim, hidden_dim, visium_dim):
        super(Xiprior, self).__init__()
        
        #self.network = nn.Embedding(k_dim, hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim,momentum=0.1, eps=1e-5),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden_dim, visium_dim)
        self.logvar_layer = nn.Linear(hidden_dim, visium_dim)

    def forward(self, z):
        #x_input = c.argmax(dim=-1)#torch.concat([c] ,dim=1) 
        #x_input = x_input.to(torch.int)
        hidden = F.relu(self.network(z))
        mu = F.softmax(self.mu_layer(hidden),dim=-1)
        log_var= self.logvar_layer(hidden)
        return mu,log_var
    
class XEncoder(nn.Module):
    """ q(x|X, c) encoding visium , and cell type c, to approixmates the posterior distribution of latent feature vector z """

    def __init__(self, codex_dim, visium_dim, visium_hidden_dim, hidden_dim, cell_niche_dim, z_dim, k_dim,num_heads=1):
        super(XEncoder, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(visium_dim+k_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim,momentum=0.1, eps=1e-5),
            nn.ReLU()
        )
        
        self.mu_layer = nn.Sequential(nn.Linear(hidden_dim+k_dim, visium_dim),
                                      nn.Softplus()
                                     )
        
    def forward(self, X, c_onehot):
        hidden = self.network(torch.cat([X,c_onehot], dim=1))          
        mu = self.mu_layer(torch.cat([hidden,c_onehot], dim=1))
        return mu#, log_var   
    
class ZEncoder(nn.Module):
    """ q(z|y,x, G) encoding codex and visium , and graph G, to approixmates the posterior distribution of latent feature vector z """

    def __init__(self, codex_dim, visium_dim, visium_hidden_dim, hidden_dim, cell_niche_dim, z_dim, k_dim,num_heads=1):
        super(ZEncoder, self).__init__()
        
        
        self.network = nn.Sequential(
            nn.Linear(codex_dim+visium_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mu_layer = nn.Linear(hidden_dim, z_dim)
        self.log_var_layer = nn.Linear(hidden_dim, z_dim)

    #def forward(self, concated_xy, graph_features_tensor):
    def forward(self, concated_xy):
        #hidden = self.network(torch.cat([concated_xy, graph_features_tensor], dim=1))  
        hidden = self.network(torch.cat([concated_xy], dim=1))  
        mu = self.mu_layer(hidden)
        log_var = self.log_var_layer(hidden)
        
        return mu, log_var
    

class VEncoder(nn.Module):
    def __init__(self, z_dim, visium_dim, visium_hidden_dim, codex_dim, v_dim, hidden_dim, cell_niche_dim, k_dim):
        super(VEncoder, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(codex_dim+visium_dim+cell_niche_dim+z_dim, hidden_dim),
            #nn.Linear(codex_dim+visium_dim+z_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim, v_dim)
        self.log_var_layer = nn.Linear(hidden_dim, v_dim)
        
    def forward(self,  concated_xy, graph_features_tensor, qz):
    #def forward(self,  concated_xy, qz):
        
        hidden = self.network(torch.cat([concated_xy, graph_features_tensor,qz], dim=1)) 
        #hidden = self.network(torch.cat([concated_xy, qz], dim=1)) 
        mu = self.mu_layer(hidden)
        log_var = self.log_var_layer(hidden)
        return mu, log_var
    
class myGATConv(GATConv):
    def __init__(self, *args, **kwargs):
        super(myGATConv, self).__init__(*args, **kwargs)
        
        self.dropout = kwargs.get('dropout', 0.6)
        self.store_attention_weights = kwargs.get('return_attention_weights', False)
        self.attention_weights = None

    def forward(self, x, edge_index, size=None, return_attention_weights=True):
        
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Call the original GAT forward method, expecting attention weights returned if configured
        node_features, (edge_indices, attention_coeffs) = super().forward(x, edge_index, size=size, return_attention_weights=True)
        
        if torch.isnan(attention_coeffs).any():
            attention_coeffs = torch.nan_to_num(attention_coeffs)
            
        attention_coeffs = F.softmax(attention_coeffs, dim=1)
        if return_attention_weights:
            return node_features, (edge_indices, attention_coeffs)
        else:
            return node_features
            
class CORAL_model(torch.nn.Module):

    def __init__(self,  args):
        super().__init__()

        self.args = args
        self.visium_dim = args['visium_dim']
        self.codex_dim = args['codex_dim']
        self.hidden_dim = args['hidden_dim']
        self.z_dim = args['latent_dim']
        self.u_dim = args['u_dim']
        self.k_dim = args['type_dim']
        self.v_dim = 1
        self.device = args['device']
        self.visium_hidden_dim = args['visium_hidden_dim']
        self.eps=1e-5
        self.cell_niche_dim = 20

        
        ### networks
        self.gat = myGATConv(in_channels=self.codex_dim, 
                             out_channels=self.cell_niche_dim, 
                             heads=1, 
                             concat=True, 
                             dropout=0.6,
                             return_attention_weights=True
                            )
        #self.gat.apply(init_weights)
       
        self.vi_net = VEncoder(z_dim=self.z_dim, visium_hidden_dim = self.visium_hidden_dim, cell_niche_dim=self.cell_niche_dim,visium_dim=self.visium_dim, codex_dim= self.codex_dim, v_dim=self.v_dim,hidden_dim=self.hidden_dim, k_dim=self.k_dim).to(self.device)
    
    
        self.zi_net = ZEncoder(visium_dim=self.visium_dim, cell_niche_dim=self.cell_niche_dim,visium_hidden_dim = self.visium_hidden_dim,codex_dim= self.codex_dim, hidden_dim=self.hidden_dim, z_dim=self.z_dim, k_dim =self.k_dim).to(self.device)
        
        self.xi_net = XEncoder(visium_dim=self.visium_dim, cell_niche_dim=self.cell_niche_dim,visium_hidden_dim = self.visium_hidden_dim,codex_dim= self.codex_dim, hidden_dim=self.hidden_dim, z_dim=self.z_dim, k_dim =self.k_dim).to(self.device)

        self.zi_prior = Zprior(k_dim=self.k_dim, u_dim=self.u_dim,hidden_dim=self.hidden_dim, z_dim=self.z_dim, v_dim=self.v_dim).to(self.device)

        self.xi_prior = Xiprior(z_dim=self.z_dim,hidden_dim=self.hidden_dim, visium_dim=self.visium_dim).to(self.device)

        self.lj_enc = nn.Sequential(
                                nn.Linear(self.visium_dim, self.hidden_dim, bias=True),
                                nn.BatchNorm1d(self.hidden_dim, momentum=0.1, eps=1e-5),
                                nn.ReLU(),
        ).to(self.device)
        
        self.lj_enc_m = nn.Linear(self.hidden_dim, 1).to(self.device)
        self.lj_enc_logv = nn.Linear(self.hidden_dim, 1).to(self.device)

        self.li_enc = nn.Sequential(
                                nn.Linear(self.codex_dim, self.hidden_dim, bias=True),
                                nn.BatchNorm1d(self.hidden_dim, momentum=0.1, eps=1e-5),
                                nn.ReLU(),
        ).to(self.device)
        
        self.li_enc_m = nn.Linear(self.hidden_dim, 1).to(self.device)
        self.li_enc_logv = nn.Linear(self.hidden_dim, 1).to(self.device)
        

        self._px_r = torch.nn.Parameter(torch.randn(self.visium_dim)*0.1,requires_grad=True)
        self._px_r_sc = torch.nn.Parameter(torch.randn(self.visium_dim)*10,requires_grad=True)
        
        self._py_r = torch.nn.Parameter(torch.randn(self.codex_dim),requires_grad=True)

        
        self.px_hidden_decoder = nn.Sequential(
                                nn.Linear(self.z_dim, self.hidden_dim, bias=True),
                                nn.BatchNorm1d(self.hidden_dim,momentum=0.1, eps=1e-5),
                                nn.ReLU(),
        ).to(self.device)
        
        self.px_scale_decoder = nn.Sequential(
                              nn.Linear(self.hidden_dim,self.visium_dim),
                              nn.Softmax(dim=-1)
        ).to(self.device)
        
        
        self.py_hidden_decoder = nn.Sequential(
                                nn.Linear(self.z_dim, self.hidden_dim, bias=True),
                                nn.BatchNorm1d(self.hidden_dim,momentum=0.1, eps=1e-5),
                                nn.ReLU()
        ).to(self.device)

        
        self.py_scale_decoder = nn.Sequential(
                                nn.Linear(self.hidden_dim,self.codex_dim),
                                nn.Softplus()
        ).to(self.device)
        
        self.visium_hidden_decoder = nn.Sequential(
                                nn.Linear(self.visium_dim,self.k_dim*self.visium_hidden_dim),
                                nn.ReLU()
        ).to(self.device)
        
        

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(mu)
        return mu + eps * std

    @property
    def px_r(self):
        return F.softplus(self._px_r) + self.eps
    @property
    def px_r_sc(self):
        return F.softplus(self._px_r_sc) + self.eps
        
    @property
    def py_r(self):
        return F.softplus(self._py_r) + self.eps

    
    def forward(self, visium_data, codex_data, cell_type, graph_data, num_ct, spot_id):
        """
        """
        #prepare the combined data [x,Y]_{i}^{p+g}
        combined_data = []
        ci = []
        spot_id_cell = []
        for visium, codex, cell_type_, spot_id_ in zip(visium_data, codex_data, cell_type, spot_id):
            C = codex.size(0)  
            if len(visium.shape) == 1:
                visium = visium.unsqueeze(0)
                
            # assum the visium is the summation of the codex 
            replicated_visium = visium.repeat(C, 1) 
            
            spot_id_cell.append(torch.Tensor(spot_id_).repeat(C, 1)) 
            concatenated = torch.cat([codex,replicated_visium], dim=1)
            combined_data.append(concatenated)  
            
            for i in cell_type_:
                ci_ = F.one_hot(i.long(), num_classes=num_ct)
                ci.append(ci_)
            
                
        ci = torch.stack(ci).to(self.device) 
        combined_data = (torch.cat(combined_data,dim=0)).to(self.device)
        
        
        qlj = torch.log1p(torch.stack(visium_data).sum(1)).unsqueeze(1).to(self.device)
        
        
        
        qli_x = []
        for i, codex in enumerate(codex_data):
            #print(C)
            C = codex.size(0)  
            qli_x.append(qlj[i].repeat(C, 1)/C)
            

        qli_x = torch.concat(qli_x)
        qli_y = torch.log1p(torch.concat(codex_data).sum(1)).unsqueeze(1).to(self.device)

        
        ## infer xi
        q_xi_m = self.xi_net(torch.log1p(combined_data[:,self.codex_dim:]).to(self.device),ci)
        q_xi_dist = NegBinom(torch.exp(qli_x)*q_xi_m+ self.eps,torch.exp(self.px_r_sc), device = self.device)
        q_xi = q_xi_dist.sample()
        
        
        #reduced_visium = ci_broadcast *self.visium_hidden_decoder(combined_data[:,self.codex_dim:].to(self.device))
        
        # xi+yi
        combined_data_new = torch.concat([torch.log1p(combined_data[:,:self.codex_dim]), torch.log1p(q_xi)],dim=1).to(self.device)
        
        
        
        
        #print(q_xi)
        
        flattened_graph_data = [item for sublist in graph_data for item in sublist]

        graph_features = []
        for data in flattened_graph_data:
            data = data.to(self.device)
            
            node_repr,(edge_indices,attention_weights) = self.gat(data.x, data.edge_index)
            
            graph_repr = global_mean_pool(node_repr, data.batch)
            graph_features.append(graph_repr)
            
        graph_features_tensor = torch.stack(graph_features).squeeze(1)
        
        
        
        
        # infer zi 
        #q_zi_m, q_zi_logvar = self.zi_net(torch.log1p(combined_data_new).to(self.device),graph_features_tensor)
        q_zi_m, q_zi_logvar = self.zi_net(combined_data_new)
        
        
        
        
        #q_zi_m, q_zi_logvar = self.zi_net(torch.log1p(combined_data+self.eps))
        q_zi = self.reparameterize(q_zi_m, q_zi_logvar)
        
        
        
        # infer vi
        q_vi_m, q_vi_logvar = self.vi_net(torch.log1p(combined_data_new).to(self.device),graph_features_tensor, q_zi)
        
        #q_vi_m, q_vi_logvar = self.vi_net(combined_data_new, q_zi)
        q_vi = self.reparameterize(q_vi_m, q_vi_logvar)
        ## generative model

        
        # p_zi
        p_zi_m,p_zi_logvar= self.zi_prior(q_vi,ci)
        p_zi = self.reparameterize(p_zi_m, p_zi_logvar)
        

        hidden = self.py_hidden_decoder(torch.cat([q_zi],dim=1))
        py_scale = self.py_scale_decoder(torch.cat([hidden],dim=1))
        py_rate  = torch.exp(qli_y) * py_scale + self.eps
        
        # generate p(xi) and p(Xj)
        hidden = self.px_hidden_decoder(torch.cat([q_zi],dim=1))
        px_scale = self.px_scale_decoder(torch.cat([hidden],dim=1))
        px_rate  = torch.exp(qli_x) * px_scale + self.eps
        
        aggregated_px_scale = []
        code_size = 0
        code_size_temp = code_size
        for _, codex in zip(visium_data, codex_data):
            code_size_temp += codex.size(0) 
            aggregated_px_scale.append(px_scale[code_size:code_size_temp,:].sum(dim=0,keepdim=False))
            code_size = code_size_temp
            
        aggregated_px_scale = torch.stack(aggregated_px_scale)
        px_rate_aggregated = torch.exp(qlj) * aggregated_px_scale + self.eps

        
        #print(px_scale)
        #print(aggregated_px_scale)
        return {
                'q_vi_m':q_vi_m,
                'q_vi_logvar':q_vi_logvar,
                'ci':ci,
                
                'p_zi_m':p_zi_m,
                'p_zi_logvar':p_zi_logvar,
                'q_zi_m':q_zi_m,
                'q_zi_logvar':q_zi_logvar,
                'q_zi':q_zi,
                
                'qli_x':qli_x,
                'qli_y':qli_y,
                'qlj':qlj,
                
                'px_r_sc':self.px_r_sc,
                'px_rate':px_rate,
                'q_xi_m':q_xi_m,
                'q_xi':q_xi,
            
                
                'px_rate_aggregated':px_rate_aggregated,
                'px_r':self.px_r,
                'aggregated_px_scale':aggregated_px_scale,
                
                'py_r':self.py_r,
                'py_rate':py_rate,
                'spot_id_cell':spot_id_cell,
                'graph_features_tensor': graph_features_tensor
               } 
        


    def efficient_contrastive_loss(self, outputs, labels, margin=50):
        """
        Efficient version of contrastive loss using vectorized operations.
        """
        # Compute pairwise distances in a vectorized way
        pairwise_dist = torch.pdist(outputs, p=2)
    
        # Get the indices of the upper triangular part excluding the diagonal
        triu_indices = torch.triu_indices(labels.size(0), labels.size(0), 1)
    
        # Get the labels for each pair
        labels_a = labels[triu_indices[0]]
        labels_b = labels[triu_indices[1]]
    
        # Check if labels are same (positive pair) or different (negative pair)
        is_positive = labels_a == labels_b
        is_negative = labels_a != labels_b
    
        # Select positive and negative distances
        positive_pairs = pairwise_dist[is_positive]
        negative_pairs = pairwise_dist[is_negative]
    
        # Calculate loss for positive and negative pairs
        positive_loss = positive_pairs.sum()
        negative_loss = F.relu(margin - negative_pairs).sum()
    
        # Normalize the loss
        n_pos = positive_pairs.size(0)
        n_neg = negative_pairs.size(0)
        loss = (positive_loss / n_pos) + (negative_loss / n_neg) if n_pos > 0 and n_neg > 0 else torch.tensor(0.0)
    
        return loss  

    def loss_function(self, output, visium_data, codex_data, codex_type, sc_rna):
        
        combined_data = []
        
        for visium, codex in zip(visium_data, codex_data):
            
            C = codex.size(0)  
            if len(visium.shape) == 1:
                visium = visium.unsqueeze(0)
            replicated_visium = visium.repeat(C, 1)  
            concatenated = torch.cat([codex, replicated_visium], dim=1)
            combined_data.append(concatenated)
            
        combined_data = torch.cat(combined_data,dim=0)
        visium_data = torch.stack(visium_data)
        codex_data = torch.concat(codex_data)

        kl_div_vi =kl( Normal( output['q_vi_m'], torch.exp(output['q_vi_logvar'] / 2)),
                       Normal( torch.zeros_like(output['q_vi_m']), torch.exp(torch.ones_like(output['q_vi_logvar']) / 2))
        ).sum(dim=1).mean()
        
        
        q_xi_dist = NegBinom(torch.exp(output['qli_x'])*output['q_xi_m']+ self.eps,torch.exp(output['px_r_sc']),device = self.device)
        q_xi = q_xi_dist.sample()#((128,)).mean(axis=0)

        xi_recon_loss  = - NegBinom(output['px_rate'], torch.exp(output['px_r_sc']),device = self.device).log_prob(q_xi).sum(-1).mean() + 0.5*self.efficient_contrastive_loss(torch.exp(output['qli_x'])*output['q_xi_m'], output['ci'].argmax(dim=-1)) 
        
        visium_recon_loss  = - NegBinom(output['px_rate_aggregated'], torch.exp(output['px_r']),device = self.device).log_prob(visium_data).sum(-1).mean()
        
        
        codex_recon_loss =  - Gamma(output['py_rate'], torch.exp(output['py_r'])).log_prob(codex_data+self.eps).sum(-1).mean()
        
        
        kl_div_zi = kl(
                    Normal( output['q_zi_m'], torch.exp(output['q_zi_logvar'] / 2)),
                    Normal( output['p_zi_m'], torch.exp(output['p_zi_logvar'] / 2))
        ).sum(dim=1).mean() 
        
        # Incorporate GAT-based graph features into the loss
        graph_feature_loss = F.mse_loss(output['graph_features_tensor'], torch.ones_like(output['graph_features_tensor']))

        
        total_loss = 1e3*visium_recon_loss + codex_recon_loss + xi_recon_loss + kl_div_zi + kl_div_vi  + graph_feature_loss
        

        
   
        return total_loss, visium_recon_loss, codex_recon_loss, kl_div_vi, kl_div_zi, xi_recon_loss,
 


