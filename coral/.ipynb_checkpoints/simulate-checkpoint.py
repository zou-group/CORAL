import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Dirichlet, constraints,  Normal,LogNormal, Gamma, Poisson
  
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

    def __init__(self, mu, theta, eps=1e-5):
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

    def sample(self,sample_shape=torch.Size()):
        lambdas = Gamma(
            concentration=self.theta + self.eps,
            rate=(self.theta + self.eps) / (self.mu + self.eps),
        ).rsample(sample_shape)

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
      
class SharedEncoder(nn.Module):
    def __init__(self, num_categories, hidden_dim, z_dim):
        super(SharedEncoder, self).__init__()
        self.num_categories=num_categories
        self.network = nn.Sequential(
            nn.Linear(num_categories, hidden_dim),
            nn.BatchNorm1d(hidden_dim,momentum=0.1, eps=1e-5),
            nn.ReLU()
        )
        
        self.mu_layer = nn.Linear(hidden_dim, z_dim)
        self.log_var_layer = nn.Linear(hidden_dim, z_dim)

    def forward(self, c):
        c_onehot = F.one_hot(c, num_classes=self.num_categories).float()
        c_embed = self.network(c_onehot)
        mu = self.mu_layer(c_embed)
        log_var = self.log_var_layer(c_embed)
        return mu, log_var

class CODEXDecoder(nn.Module):
    def __init__(self, z_dim, num_categories, hidden_dim, protein_dim):
        super(CODEXDecoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim,momentum=0.1, eps=1e-5),
            nn.ReLU(),
        )
        self.mu_layer =  nn.Sequential(
            nn.Linear(hidden_dim, protein_dim),
            nn.Softplus()
        )
        self.py_r = F.softplus(torch.nn.Parameter(torch.randn(protein_dim)+5,requires_grad=False))

    def forward(self, z):
        hidden = self.network(z)
        mu = self.mu_layer(hidden)
        return mu,self.py_r

    def sample(self, z):
        mu, log_var = self(z)
        codex = Gamma(mu*1e3, torch.exp(log_var)).sample()
        #codex = F.relu(codex)
        return codex
        
class VisiumDecoder(nn.Module):
    def __init__(self, z_dim, num_categories, hidden_dim, gene_dim):
        super(VisiumDecoder, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim,momentum=0.1, eps=1e-5),
            nn.ReLU(),
            nn.Linear(hidden_dim, gene_dim),
            nn.Softmax(dim=-1)
        )
        self.px_r = F.softplus(torch.nn.Parameter(torch.randn(gene_dim)+5,requires_grad=False))

    def forward(self, z):
        x_input = torch.cat([z], dim=1)
        mu = self.network(x_input)
        return mu, self.px_r





