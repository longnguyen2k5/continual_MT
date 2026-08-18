import torch 
import torch.nn as nn 
from core.base_adapter import ContinualAdapter
from core.components import StackedLoRAExperts
import torch.nn.functional as F

class OMoELinear(ContinualAdapter): 
    def __init__(self, base_layer: nn.Linear, r: int=16, lora_alpha: int=1, init_strategy: str= 'kaiming', num_omoe_experts: int = 2): 
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        self.r = r
        self.scaling = lora_alpha / torch.sqrt(torch.tensor(r, dtype=torch.float32))
        self.target_dtype = base_layer.weight.dtype
        self.num_experts = num_omoe_experts
        
        self.init_strategy = init_strategy
        self.register_buffer('num_reset', torch.tensor(0, dtype=torch.long))
        self.weight = nn.Parameter(base_layer.weight.data, requires_grad=False)
        if base_layer.bias is not None: 
            self.bias = nn.Parameter(base_layer.bias.data, requires_grad=False)
        else: 
            self.register_buffer('bias', None)
        
        self.lora_experts = StackedLoRAExperts(r=r, lora_alpha=lora_alpha, 
                                                in_features=self.in_features, 
                                                out_features=self.out_features, 
                                                num_experts=self.num_experts, 
                                                init_strategy=self.init_strategy)
        
        self.router = nn.Linear(self.in_features, self.num_experts)
    
    def get_whitelist_keys(self):
        return [
            'num_reset', 
            'lora_experts',
            'router'
        ]
    
    
    
    def forward(self, x: torch.Tensor): 
        batch_size, seq_len, _ = x.shape
        base_out = F.linear(x, self.weight, self.bias) 
        
        router_weight = self.router.weight.to(x.dtype)
        logits = F.linear(x, router_weight)
        experts_mask = torch.softmax(logits, dim=-1, dtype=torch.float32).to(x.dtype) # shape: (batch_size, seq_len, num_experts)
        experts_output = self.lora_experts(x) # shape: (batch_size, seq_len, num_experts, out_features)
        
        E_matrix = experts_output.transpose(-1, -2)
        Q, R = torch.linalg.qr(E_matrix.float()) 
        orthogonalized_output = Q.transpose(-1, -2).to(x.dtype)
        
        moe_out = torch.einsum('bseo, bse -> bso', orthogonalized_output, experts_mask) # shape: (batch_size, seq_len, out_features)
        
        return base_out + moe_out

    