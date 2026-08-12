import torch
import math 
import torch.nn as nn 
import torch.nn.functional as F
from base_adapter import ContinualAdapter
class LoRABase(ContinualAdapter): 
    def __init__(self, base_layer: nn.Linear, r: int=16, lora_alpha: int=1, init_strategy: str= 'kaiming'): 
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        self.r = r
        self.scaling = lora_alpha / math.sqrt(r)
        self.target_dtype = base_layer.weight.dtype
        
        self.init_strategy = init_strategy
        self.register_buffer('num_reset', torch.tensor(0, dtype=torch.long))
        self.weight = nn.Parameter(base_layer.weight.data, requires_grad=False)
        if base_layer.bias is not None: 
            self.bias = nn.Parameter(base_layer.bias.data, requires_grad=False)
        else: 
            self.register_buffer('bias', None)
        
        self.A_curr = nn.Parameter(torch.empty(self.r, self.in_features))
        self.B_curr = nn.Parameter(torch.empty(self.out_features, self.r))
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
                weight_fp32 = self.weight.data.to(torch.float32)
                U, S, V = torch.svd_lowrank(weight_fp32, q=self.r, niter=4)
                self.A_curr.data = V.t()
                nn.init.zeros_(self.B_curr)
        elif self.init_strategy == 'pissa': 
            if self.num_reset.item() == 0:
                with torch.no_grad(): 
                    weight_fp32 = self.weight.data.to(torch.float32)
                    U, S, V = torch.svd_lowrank(weight_fp32, q=self.r, niter=4)
                    S_diag = torch.diag(S) / self.scaling
                    sqrt_S = torch.sqrt(S_diag)
                    
                    self.A_curr.data = torch.mm(sqrt_S, V.t())
                    self.B_curr.data = torch.mm(U, sqrt_S)
                    self.register_buffer('A_core', self.A_curr.data.clone())
                    self.register_buffer('B_core', self.B_curr.data.clone())
                    
                    W_core_fp32 = torch.mm(self.B_core, self.A_core) * self.scaling
                    self.weight.data = self.weight.data - W_core_fp32.to(self.target_dtype)
            else: 
                with torch.no_grad():
                    self.A_curr.data = self.A_core.clone()
                    self.B_curr.data = self.B_core.clone()

        else: 
            raise ValueError(f"Unknown init_strategy: {self.init_strategy}")
        
        self.num_reset += 1
        
    def compute_delta_w(self): 
        delta_w = torch.mm(self.B_curr, self.A_curr) * self.scaling
        return delta_w.to(self.target_dtype)

    def pre_load_undo(self): 
        if self.init_strategy == 'pissa': 
            with torch.no_grad(): 
                W_core_trash = torch.mm(self.B_core, self.A_core) * self.scaling
                self.weight.data = self.weight.data + W_core_trash
    
    def post_load_redo(self): 
        if self.init_strategy == 'pissa': 
            with torch.no_grad(): 
                W_core_loaded = torch.mm(self.B_core, self.A_core) * self.scaling
                self.weight.data = self.weight.data - W_core_loaded
                
class StandardLoRALinear(LoRABase):
    def __init__(self, base_layer: nn.Linear, r: int=16, lora_alpha: int=1, init_strategy: str='kaiming'):
        super().__init__(base_layer, r=r, lora_alpha=lora_alpha, init_strategy=init_strategy)
    
    def get_whitelist_keys(self) -> list:
        return [
            'A_curr',
            'B_curr',
            'A_core',
            'B_core',
            'num_reset'
        ]

    def forward(self, x: torch.Tensor): 
        delta_w = self.compute_delta_w() # out_features * in_features
        w_active = self.weight + delta_w # out_features * in_features
        return F.linear(x, w_active, self.bias)