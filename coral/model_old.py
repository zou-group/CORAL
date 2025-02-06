import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch_geometric.nn import SGConv
from torch.distributions import kl_divergence as kl
from torch.distributions import constraints, Distribution, Normal, Gamma, Poisson,Categorical,LogNormal
from torch.distributions import NegativeBinomial



class PriorZGivenC(nn.Module):
    def __init__(self, type_dim, hidden_dim, z_dim):
        super(PriorZGivenC, self).__init__()
        self.embedding = nn.Embedding(type_dim, hidden_dim)
        self.mu_layer = nn.Linear(hidden_dim, z_dim)
        self.log_var_layer = nn.Linear(hidden_dim, z_dim)

    def forward(self, c):
        c_embed = self.embedding(c)
        mu = self.mu_layer(c_embed)
        log_var = self.log_var_layer(c_embed)
        return mu, log_var
    
class ZEncoder(nn.Module):
    def __init__(self, codex_dim, visium_dim, k_dim, hidden_dim, z_dim):
        super(ZEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(codex_dim+visium_dim +k_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim, z_dim)
        self.log_var_layer = nn.Linear(hidden_dim, z_dim)

    def forward(self, yi,Xi,c):
        
        x_input = torch.cat([yi, Xi, c], dim=1)
        hidden = self.network(x_input)
        mu = self.mu_layer(hidden)
        log_var = self.log_var_layer(hidden)
        return mu, log_var
    
class CEncoder(nn.Module):
    def __init__(self, codex_dim, visium_dim, hidden_dim, k_dim):
        super(CEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(codex_dim+visium_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, k_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, yi,Xi):
        x_input = torch.cat([yi, Xi], dim=1)
        q_pi = self.network(x_input)
        return q_pi

class CODEXDecoder(nn.Module):
    def __init__(self, z_dim, type_dim, hidden_dim, protein_dim):
        super(CODEXDecoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(z_dim + type_dim, hidden_dim),
            #nn.ReLU(),
            nn.Linear(hidden_dim, protein_dim),
            nn.Sigmoid()
        )
        self.type_dim = type_dim

    def forward(self, z, c):
        x_input = torch.cat([z, c], dim=1)
        mean = self.network(x_input)
        # assuming a shared dispersion for simplicity
        #protein_dispersion = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        
        return mean#, protein_dispersion
    
        

class VisiumDecoder(nn.Module):
    def __init__(self, z_dim, type_dim, hidden_dim, gene_dim):
        super(VisiumDecoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(z_dim + type_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim,momentum=0.01, eps=0.001),
            nn.ReLU(),
            #nn.Linear(hidden_dim, gene_dim),
            #nn.Tanh()
            #nn.Softmax(dim=-1),
        )
        self.mu_layer = nn.Sequential(nn.Linear(hidden_dim, gene_dim),
                                      nn.Tanh()
        )
        self.log_var_layer = nn.Sequential(nn.Linear(hidden_dim, gene_dim),
                                      nn.Tanh()
        )
        self.type_dim = type_dim

    def forward(self, z, c):
        x_input = torch.cat([z, c], dim=1)
        prob = self.network(x_input)
        mu = self.mu_layer(prob)
        log_var = self.log_var_layer(prob)
        return mu, log_var
        # assuming a shared dispersion for simplicity
        #gene_dispersion = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        #return prob, gene_dispersion



class CORAL_model(torch.nn.Module):

    def __init__(self,  args):
        super().__init__()

        self.args = args
        self.visium_dim = args['visium_dim']
        self.codex_dim = args['codex_dim']
        self.hidden_dim = args['hidden_dim']
        self.z_dim = args['latent_dim']
        self.k_dim = args['type_dim']
        self.device = args['device']
        self.eps=1e-5

        self.prior_z_given_c = PriorZGivenC(type_dim=self.k_dim, hidden_dim=self.hidden_dim, z_dim=self.z_dim)

        self.codex_decoder = CODEXDecoder(z_dim=self.z_dim, type_dim=self.k_dim, hidden_dim=self.hidden_dim, protein_dim=self.codex_dim)
        
        self.visium_decoder = VisiumDecoder(z_dim=self.z_dim, type_dim=self.k_dim, hidden_dim=self.hidden_dim, gene_dim=self.visium_dim)

        self.ci_net = CEncoder(codex_dim=self.codex_dim, visium_dim=self.visium_dim, hidden_dim=self.hidden_dim, k_dim=self.k_dim)

        self.zi_net = ZEncoder(codex_dim=self.codex_dim, visium_dim=self.visium_dim, 
                               k_dim = self.k_dim, hidden_dim=self.hidden_dim, z_dim=self.z_dim)


    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(mu)
        return mu + eps * std
        
    def gumbel_softmax(self, logits, temperature=1.0):
        # Draw Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        sampled_values = (logits + gumbel_noise) / temperature
        return F.softmax(sampled_values, dim=-1)
    
    
    def forward(self, visium_data, codex_data):

        batch_size = len(visium_data)
        
        
        codex_num = codex_data[0].shape[0]
        
        codex_data = codex_data[0]
        visium_data = visium_data[0].unsqueeze(0)
        
        replicated_visium = visium_data.expand(codex_num, -1)
        
        q_ci = self.ci_net(codex_data, replicated_visium)
        
        c_onehot = F.one_hot(q_ci.argmax(dim=-1), num_classes=self.k_dim).float()
        
        q_zi_m, q_zi_logvar = self.zi_net(codex_data, replicated_visium, c_onehot)
        #print(q_zi_m)
        q_zi = self.reparameterize(q_zi_m, q_zi_logvar)
        
        p_yi = self.codex_decoder(q_zi, c_onehot)

        p_xi_p, p_xi_r = self.visium_decoder(q_zi, c_onehot)

        Xj_aggregated = p_xi_p.sum(dim=0,keepdim=False) 
        Xj_aggregated = Xj_aggregated.unsqueeze(0)
    
        return {'q_ci':q_ci,
                'q_zi_m':q_zi_m,
                'q_zi_logvar':q_zi_logvar,
                'p_xi_p':p_xi_p, 
                'p_xi_r':p_xi_r,
                'p_yi':p_yi,
                'Xj_aggregated':Xj_aggregated,
                'q_zi_m':q_zi_m, 
                'q_zi_logvar':q_zi_logvar
               }

    

    def loss_function(self, output, batch):
        # Extracting the necessary parameters from the output
        
        codex_data = batch['codex'][0]
        visium_data = batch['visium'][0].unsqueeze(0)
        # Normalize Codex data
        #codex_data = (codex_data - codex_data.mean(axis=0)) / codex_data.std(axis=0)

        codex_recon_loss = F.mse_loss(output['p_yi'], codex_data, reduction='sum')  # Single-cell level codex data

        visium_recon_loss =torch.tensor(0.0)#F.mse_loss(output['Xj_aggregated'], visium_data, reduction='mean')  # Single-cell level codex data
        #visium_recon_loss =kl(Normal(output['Xj_aggregated'], torch.ones_like(output['Xj_aggregated'])),
        #               Normal(visium_data, torch.ones_like(output['Xj_aggregated']))).sum(dim=1).mean()
        
        # Parameters for the prior Log-Normal distribution
        #pxi_mean = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)  # Initializing with 0 for the underlying normal
        #pxi_std = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)   # A positive value for std of the underlying normal
        # Compute KL divergence directly
        kl_div_xi = torch.tensor(0.0)#kl(Normal(torch.zeros_like(output['p_xi_p']), torch.ones_like(output['p_xi_r'])),
                    #   Normal(output['p_xi_p'], torch.ones_like(output['p_xi_r'])),).sum(dim=1).mean()
        
        #loss = nn.PoissonNLLLoss(reduction='mean')
        #log_input = output['Xj_aggregated']
        #target = visium_data
        #visium_recon_loss = loss(log_input, target)

        pc_dist = Categorical(probs=torch.tensor([1/self.k_dim]*self.k_dim))
        kl_div_ci = torch.tensor(0.0)#kl(Categorical(probs=output['q_ci']), pc_dist).sum()
        
        #p_zi_mean,p_zi_logvar = self.prior_z_given_c(pc_dist.sample())
        kl_div_zi = kl(Normal(output['q_zi_m'], torch.exp(output['q_zi_logvar'] / 2)),
                       #Normal(p_zi_mean, torch.exp(p_zi_logvar / 2))
                      Normal(torch.zeros_like(output['q_zi_m']), torch.ones_like(output['q_zi_logvar']))
                       ).sum(dim=1).mean()
        
        #visium_dist = NegBinom(mu=output['p_xi_p'], theta=output['p_xi_r'])
        #visium_libarary = visium_data.sum(dim=1,keepdim=True)
        #pxi_r = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        #pxi_p = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        #pxi_dist = NegBinom(mu=pxi_p, theta=pxi_r)
        #print(output['p_xi_p'])
        
        #samples = visium_dist.sample()
        #kl_div_xi = (visium_dist.log_prob(samples) - pxi_dist.log_prob(samples)).mean()

        total_loss = (visium_recon_loss 
                       + codex_recon_loss 
                       + kl_div_zi 
                       + kl_div_ci 
                       + kl_div_xi 
                       )
        
        return total_loss, visium_recon_loss, codex_recon_loss, kl_div_zi, kl_div_ci, kl_div_xi



# Reference:
# https://github.com/YosefLab/scvi-tools/blob/master/scvi/distributions/_negative_binomial.py
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

    def __init__(self, mu, theta, eps=1e-10):
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
        self.mu = mu
        self.theta = theta
        self.eps = eps
        super(NegBinom, self).__init__(validate_args=True)

    def sample(self):
        lambdas = Gamma(
            concentration=self.theta + self.eps,
            rate=(self.theta + self.eps) / (self.mu + self.eps),
        ).rsample()

        x = Poisson(lambdas).sample()

        return x

    def log_prob(self, x):
        """log-likelihood"""
        ll = torch.lgamma(x + self.theta) - \
             torch.lgamma(x + 1) - \
             torch.lgamma(self.theta) + \
             self.theta * (torch.log(self.theta + self.eps) - torch.log(self.theta + self.mu + self.eps)) + \
             x * (torch.log(self.mu + self.eps) - torch.log(self.theta + self.mu + self.eps))

        return ll