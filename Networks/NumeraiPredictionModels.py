### Initialize the Neural Networks
import torch as th
import torch.nn as nn
import os
"""
let x = Features Batch
x -> let opinions = {ResidualFeatureEncoder_i(x) for ResidualFeatureEncoder in self.Experts}
 -> let consensus = Sum(opinions) -> Decoder(consensus) \eq y (Target)

"""

class ResidualBlock(nn.Module):
    def __init__(self, residual_dim=512, brodcast_dim=1048, dropout_prob=0.1, activation_fnc=nn.GELU, device='cuda'):
        super(ResidualBlock, self).__init__()
        self.device = device
        self.to(self.device)
        self.residual_block = nn.Sequential(
            nn.Linear(residual_dim, brodcast_dim),
            activation_fnc(),
            nn.Dropout(dropout_prob),
            nn.Linear(brodcast_dim, brodcast_dim),
            activation_fnc(),
            nn.Dropout(dropout_prob),
            nn.Linear(brodcast_dim, brodcast_dim),
            activation_fnc(),
            nn.Dropout(dropout_prob),
            nn.Linear(brodcast_dim, residual_dim),
            activation_fnc(), # Needs Batch Norm
        )
        self._norm = nn.LayerNorm(residual_dim)
    def forward(self, x):
        return self._norm(x + self.residual_block(x))
    
class ResidualFeatureEncoder(nn.Module):
    def __init__(self, expert_num, num_residuals=5, input_dim=313, output_dim= 256, residual_dim=512, dataset_name="Dataset_563", device='cuda'):
        super(ResidualFeatureEncoder, self).__init__()
        self.device = device
        self.to(self.device)
        self.brodcast = nn.Sequential(
            nn.Linear(input_dim, residual_dim),
            nn.GELU(),
        )
        self.residuals = [ResidualBlock(residual_dim=residual_dim,) for _ in range(num_residuals)]
        self.residuals = nn.Sequential(*self.residuals)
        self.outcast = nn.Linear(residual_dim, output_dim)

        self._expert_num = expert_num
        self._ds_name = dataset_name
        self.save_path = os.path.join('.', f'Networks','Checkpoints','NumeraiExperts', f"{self._expert_num}")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
    def forward(self, x):
        x = self.brodcast(x)
        x = self.residuals(x)
        x = self.outcast(x)
        return x
    
    def load_checkpoint(self):
        if os.path.exists(os.path.join(self.save_path, f'{self._ds_name}_{self._expert_num}_weights.pt')):
            self.load_state_dict(th.load(os.path.join(self.save_path, f'{self._ds_name}_{self._expert_num}_weights.pt')))

    
    def save_checkpoint(self, file=None):
        print(f'[Expert {self._expert_num}] Saving Checkpoint...')
        if file != None:
            th.save(self.state_dict(), file)
        else:
            th.save(self.state_dict(), self.save_path + "/" + f'{self._ds_name}_{self._expert_num}_weights.pt') 

class ExpertDecoder(nn.Module):
    def __init__(self, num_experts = 15, num_residuals=5, input_dim=310, output_dim=5, residual_input_dim=313, residual_output_dim= 256, residual_dim=512, dropout_probs=0.05, load_default=False, dataset_name="Dataset_563", device='cuda'):
        super(ExpertDecoder, self).__init__()
        self.device = device
        self.to(self.device)
        self.num_experts, self.num_residuals = num_experts, num_residuals
        self.experts = nn.ModuleList([ResidualFeatureEncoder(expert_num=expert_num, num_residuals=num_residuals, input_dim=residual_input_dim, output_dim=residual_output_dim, residual_dim=residual_dim) for expert_num in range(num_experts)])
        self._norm = nn.LayerNorm(residual_output_dim)
        self.decoder = nn.Sequential(
            nn.Linear(residual_output_dim, residual_output_dim),
            nn.GELU(),
            nn.Dropout(dropout_probs),
            nn.Linear(residual_output_dim, residual_output_dim),
            nn.GELU(),
            nn.Dropout(dropout_probs),
            nn.Linear(residual_output_dim, residual_output_dim//4),
            nn.GELU(),
            nn.Linear(residual_output_dim//4, residual_output_dim//8),
            nn.GELU(),
            nn.Linear(residual_output_dim//8, output_dim), # output 
            nn.Sigmoid(),
        )
        self._ds_name = dataset_name
        self.save_path = os.path.join('.', f'Networks','Checkpoints','NumeraiExperts', 'ExpertDecoder')
        self._file_name = f'expert_decoder_{self.num_experts}#Experts_{self.num_residuals}#Residuals_{self._ds_name}.pt'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    
        elif load_default:
            self.load_checkpoint()
            # for expert in self.experts:
            #     expert.load_checkpoint()

    def forward(self, x):
        expert_opinions = [expert(x) for expert in self.experts]
        expert_opinions_tnsr = th.stack(expert_opinions, dim=0)
        expert_consensus = th.sum(expert_opinions_tnsr, dim=0)
        normalized_consensus = self._norm(expert_consensus)
        y = self.decoder(normalized_consensus)
        return y

    def load_checkpoint(self):
        if os.path.exists(os.path.join(self.save_path, self.file_name)):
            self.load_state_dict(th.load(os.path.join(self.save_path, self.file_name)))
    
    def save_checkpoint(self, file=None):
        print(f'[Expert {self._expert_num}] Saving Checkpoint...')
        if file != None:
            th.save(self.state_dict(), file)
        else:
            th.save(self.state_dict(), self.save_path + "/" + self.file_name)

# # Testing Network
# input_size = 313
# batch_size = 3
# expert_decoder = ExpertDecoder(num_experts=8, num_residuals=8)
# print(f'Model\'s Parameter Count w/ {expert_decoder.num_experts} Experts and {expert_decoder.num_residuals} Residuals Each:',sum(p.numel() for p in expert_decoder.parameters()))
# expert_decoder.eval()
# x = th.rand(size=(batch_size, input_size))
# y = expert_decoder(x)
# print(y)   