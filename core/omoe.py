import torch 
import torch.nn as nn 
from core.base_adapter import ContinualAdapter
from core.components import StackedLoRAExperts

class OMoE(ContinualAdapter): 
    def __init__(self, base_layer: nn.Linear, r: int=16, lora_alpha: int=1, init_strategy: str= 'kaiming', num_experts: int=4): 
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        self.r = r
        self.scaling = lora_alpha / torch.sqrt(torch.tensor(r, dtype=torch.float32))
        self.target_dtype = base_layer.weight.dtype
        self.num_experts = num_experts
        
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
                                                num_experts=num_experts, 
                                                init_strategy=self.init_strategy)
        