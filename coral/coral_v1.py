import torch
import torch.optim as optim
import torch.nn as nn
from .model import CORAL_model
from typing import Dict, Tuple, List
import logging
import numpy as np
from torch.distributions import Distribution,constraints,  Normal, LogNormal,Gamma, Poisson, Categorical, kl_divergence as kl
import random

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


class CORAL:
    """Class definition for CORAL"""
    DEFAULT_LR = 1e-3
    SCHEDULER_GAMMA = 0.98

    def __init__(self, 
                 vc_data,
                 trainloader, 
                 device: str = 'cpu',
                 hidden_dim: int = 128,
                 latent_dim: int = 64,
                 type_dim: int = 5,
                 u_dim:int = 10,
                 visium_hidden_dim: int = 10,
                 gene_list = None,
                 protein_list = None,
                 visium_load = None,
                 codex_load = None,
                ) -> None:
        self.gene_list =gene_list
        self.protein_list =protein_list
        self.device = device
        self.trainloader = trainloader
        self.visium_load = visium_load
        self.codex_load = codex_load
        self.args: Dict[str, int] = {
            'hidden_dim': hidden_dim,
            'latent_dim': latent_dim,
            'u_dim':u_dim,
            'type_dim': type_dim,
            'visium_dim': len(vc_data.__getitem__(0)['visium']),
            'codex_dim': vc_data.__getitem__(0)['codex'].shape[1],
            'device': self.device, 
            'visium_hidden_dim': visium_hidden_dim,
            'num_ct': vc_data.datasets[0].cell_type.astype(int).max() - vc_data.datasets[0].cell_type.astype(int).min() + 1 
        }

        self.model = CORAL_model(self.args).to(self.device)
    
    def _model_initialize(self) -> None:
        """Initializes model parameters."""
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def save_model(self, epoch: int) -> None:
        """Saves the model state."""
        torch.save(self.model.state_dict(), f'coral_model_epoch_{epoch}.pth')
        
    def train(self,
              epochs: int = 200,
              lr: float = DEFAULT_LR
             ) -> Tuple[List[float], ...]:
        
        logging.basicConfig(level=logging.INFO)
        
        total_loss_all: List[float] = [] 
        visium_recon_loss_all: List[float] = [] 
        codex_recon_loss_all: List[float] = [] 
        kl_div_zi_all: List[float] = [] 
        kl_div_vi_all: List[float] = [] 
        kl_div_xi_all: List[float] = [] 
        kl_div_li_all: List[float] = [] 
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        max_grad_norm = 1 
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.SCHEDULER_GAMMA)
        
        for epoch in range(epochs):
            self.model.train()
            
            for step, batch in enumerate(self.trainloader):
                
                visium_batch = batch['visium']
                codex_batch = batch['codex']
                cell_type = batch['cell_type']
                spot_loc = batch['spot_loc']
                cell_loc = batch['cell_loc']
                sc_rna = batch['sc_rna']
                cell_id = batch['cell_id']
                sample_id = batch['sample_id']
                neighbors = batch['neighbors']
                graph_data = batch['graph_data']
                spot_id = batch['spot_id']
                
                
                output = self.model(visium_batch, codex_batch,cell_type,graph_data, self.args['num_ct'],spot_id)
                
                (total_loss, 
                visium_recon_loss, 
                codex_recon_loss, 
                kl_div_vi,
                kl_div_zi, 
                kl_div_xi, 
                ) = self.model.loss_function(output,
                                                     visium_batch,                    
                                                     codex_batch,
                                                     cell_type,
                                                     sc_rna
                                                    )
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()

            if (epoch + 1) % 1 == 0:
                logging.info(f"Epoch[{epoch + 1}/{epochs}], Total loss: {total_loss:.4f}, Visium recon loss: {visium_recon_loss:.4f}, Codex recon loss: {codex_recon_loss:.4f}, KL div vi: {kl_div_vi:.4f}, KL div zi: {kl_div_zi:.4f},KL div xi: {kl_div_xi:.4f}")
            
            total_loss_all.append(total_loss)
            visium_recon_loss_all.append(visium_recon_loss)
            codex_recon_loss_all.append(codex_recon_loss) 
            kl_div_zi_all.append(kl_div_zi)
            kl_div_vi_all.append(kl_div_vi)
            kl_div_xi_all.append(kl_div_xi)
            #kl_div_ui_all.append(kl_div_ui.detach().numpy())
            #kl_div_li_all.append(kl_div_l().detach().numpy())
            
            scheduler.step()
        
        return (total_loss_all,visium_recon_loss_all,codex_recon_loss_all,kl_div_vi_all,kl_div_zi_all,kl_div_xi_all)
                
            
    def evaluate(self, dataloader) -> None:
        """Evaluates the model on the provided data loader."""
        self.model.eval()  # Set the model to evaluation mode
        
        # Initialize lists to store evaluation data
        self.all_actual_visium = []
        self.all_reconstructed_visium = []
        self.all_actual_codex = []
        self.all_actual_type = []
        self.all_actual_scRNA = []
        self.all_reconstructed_codex = []
        self.all_zi_values = []
        self.all_vi_values = []
        self.all_codex_type = []
        
        self.all_cell_id = []
        self.all_sample_id = []
        

        self.all_cell_xy = []
        self.all_spot_xy = []

        self.eval_px_r = []
        self.eval_px_r_sc = [] 
        self.eval_px_rate = []
        self.eval_pz_m = []
        self.eval_pz_logv = []
        
        self.eval_qli = []
        self.eval_qlj = []
        self.eval_qlj_m = []
        self.eval_px = []
        self.eval_py = []
        self.eval_px_aggregated = []
        self.eval_px_rate_aggregated = []
        self.eval_pxi_m=[]
        self.eval_cell_id = []
        self.spot_id_cell = []
        
        
        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                
                visium_batch = batch['visium']
                codex_batch = batch['codex']
                cell_type = batch['cell_type']
                spot_loc = batch['spot_loc']
                cell_loc = batch['cell_loc']
                sc_rna = batch['sc_rna']
                cell_id = batch['cell_id']
                sample_id = batch['sample_id']
                neighbors = batch['neighbors']
                graph_data = batch['graph_data']
                spot_id = batch['spot_id']
                
                output = self.model(visium_batch, codex_batch,cell_type,graph_data, self.args['num_ct'],spot_id)
                

                final_data = []
        
                for visium, codex in zip(visium_batch, codex_batch):
                    
                    C = codex.size(0)  # Number of rows in codex data
                    if len(visium.shape) == 1:
                        visium = visium.unsqueeze(0)
                    # Replicate visium data C times
                    replicated_visium = visium.repeat(C, 1)  # Shape will be [C, G]
                    # Concatenate along the feature dimension to get [C, P+G]
                    concatenated = torch.cat([codex, replicated_visium], dim=1)
                    # Add to final list
                    final_data.append(concatenated)
        
                
                final_data = torch.cat(final_data,dim=0)
                
            
                visium_data = torch.stack(visium_batch)
                codex_data = final_data[:,:self.args['codex_dim']]
                spot_xy = torch.concat(batch['spot_loc'],dim=0)
                cell_xy = torch.concat(batch['cell_loc'],dim=0)
                cell_id = torch.concat(batch['cell_id'],dim=0)
                sample_id = torch.concat(batch['sample_id'],dim=0)
                codex_type = torch.concat(batch['cell_type'],dim=0)
                
                
                self.all_spot_xy.extend(list(spot_xy.numpy()))
                self.all_spot_xy.extend(list(spot_xy.numpy()))
                
                self.all_cell_xy.extend(list(cell_xy.numpy()))
                self.all_cell_id.extend(list(cell_id.numpy()))
                self.all_sample_id.extend(list(sample_id.numpy()))
                self.all_codex_type.extend(list(codex_type.numpy()))
                self.all_actual_visium.extend(visium_data.cpu().detach().numpy())
      
                #if  sc_rna!=None:
                    #self.all_actual_scRNA.extend(torch.concat(sc_rna).detach().numpy())
                self.all_actual_codex.extend(codex_data.cpu().detach().numpy())
                #self.all_actual_type.extend(torch.concat(batch['codex_type'],dim=0).detach().numpy())
    
                self.all_zi_values.extend(output['q_zi'].cpu().detach().numpy())
                self.all_vi_values.extend(output['q_vi_m'].cpu().detach().numpy())
            
                self.eval_px_rate.extend(output["px_rate"].cpu().detach().numpy())
                self.eval_px_rate_aggregated.extend(output["px_rate_aggregated"].cpu().detach().numpy())
                
                self.eval_pz_m.extend(output["p_zi_m"].cpu().detach().numpy())
                self.eval_pz_logv.extend(output["p_zi_logvar"].cpu().detach().numpy())
                
                self.spot_id_cell.append(output['spot_id_cell'])
            
            
                self.eval_py.extend(Gamma(output["py_rate"], torch.exp(output["py_r"])).sample().cpu().detach().numpy())
                
                self.eval_px.extend(NegBinom(torch.exp(output['qli_x'])*output['q_xi_m'],torch.exp(output['px_r_sc']),device=self.device).sample().cpu().detach().numpy())
                #self.eval_px.extend(NegBinom(output['px_rate'], torch.exp(output['px_r_sc'])).sample().cpu().detach().numpy())
                
                self.eval_px_aggregated.extend(NegBinom(output["px_rate_aggregated"], torch.exp(output["px_r"]),device=self.device).sample().cpu().detach().numpy())
                
                
   
        self.all_actual_codex = np.array(self.all_actual_codex)
        self.all_cell_id = np.array(self.all_cell_id)
        self.all_sample_id = np.array(self.all_sample_id)
        self.all_codex_type = np.array(self.all_codex_type)
        
        self.all_reconstructed_codex = np.array(self.all_reconstructed_codex)
        self.all_zi_values = np.array(self.all_zi_values)
        self.all_vi_values = np.array(self.all_vi_values)
        self.all_actual_scRNA = np.array(self.all_actual_scRNA)
        self.all_actual_type = np.array(self.all_actual_type)
        #self.spot_id_cell = np.array(self.spot_id_cell)
  
    
        
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
            mean of NegBinom. distribution
            shape - [# genes,]

        theta : torch.Tensor
            dispersion of NegBinom. distribution
            shape - [# genes,]
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