import torch
import math 
import torch.nn as nn 
import torch.nn.functional as F

class LoRABase(nn.Module): 
    def __init__(self, base_layer: nn.Linear, r: int=16, lora_alpha: int=1, init_strategy: str= 'kaiming'): 
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        self.r = r
        self.scaling = lora_alpha / math.sqrt(r)
        self.target_dtype = base_layer.weight.dtype
        
        self.init_strategy = init_strategy
        self.num_reset = 0
        self.weight = nn.Parameter(base_layer.weight.data, requires_grad=False)
        if base_layer.bias is not None: 
            self.bias = nn.Parameter(base_layer.bias.data, requires_grad=False)
        else: 
            self.register_buffer('bias', None)
        
        self.A_curr = nn.Parameter(torch.empty(self.r, self.in_features, dtype=self.target_dtype))
        self.B_curr = nn.Parameter(torch.empty(self.out_features, self.r, dtype=self.target_dtype))
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.init_strategy == 'kaiming': 
            nn.init.kaiming_uniform_(self.A_curr, a=math.sqrt(5))
            nn.init.zeros_(self.B_curr)
        elif self.init_strategy == 'normal':
            nn.init.normal_(self.A_curr, mean=0.0, std=0.02)
            nn.init.zeros_(self.B_curr)
        elif self.init_strategy == 'svd': 
            with torch.no_grad(): 
                U, S, V = torch.svd_lowrank(self.weight.data, q=self.r, niter=2)
                self.A_curr.data = V.t()
                nn.init.zeros_(self.B_curr)
        elif self.init_strategy == 'pissa': 
            with torch.no_grad(): 
                U, S, V = torch.svd_lowrank(self.weight.data, q=self.r, niter=2)
                S_diag = torch.diag(S)
                sqrt_S = torch.sqrt(S_diag)
                
                self.A_curr.data = torch.mm(sqrt_S, V.t())
                self.B_curr.data = torch.mm(U, sqrt_S)
                
                W_core = torch.mm(U, torch.mm(S_diag, V.t()))
                if self.num_reset == 0: 
                    self.weight.data = self.weight.data - W_core
        else: 
            raise ValueError(f"Unknown init_strategy: {self.init_strategy}")
        
    def compute_delta_w(self): 
        return torch.mm(self.B_curr, self.A_curr) * self.scaling
    
 
        