import torch 
import torch.nn as nn
import math

class StackedLoRAExperts(nn.Module): 
    def __init__(self, r: int, lora_alpha: int, in_features: int, out_features: int, num_experts: int, init_strategy: str='kaiming'): 
        super().__init__()
        self.num_experts = num_experts
        self.scaling = lora_alpha / math.sqrt(r)
        
        self.A_stacked = nn.Parameter(torch.empty(num_experts, in_features, r))
        self.B_stacked = nn.Parameter(torch.empty(num_experts, r, out_features)) 
    
        self.reset_parameters(init_strategy)
    
    def reset_parameters(self, init_strategy: str):
        if init_strategy == 'kaiming': 
            nn.init.kaiming_uniform_(self.A_stacked, a=math.sqrt(5))
            nn.init.zeros_(self.B_stacked)
        elif init_strategy == 'normal':
            nn.init.normal_(self.A_stacked, mean=0.0, std=0.02)
            nn.init.zeros_(self.B_stacked)
        else: 
            raise ValueError(f"Chiến lược khởi tạo '{init_strategy}' không hợp lệ. Vui lòng chọn từ ['kaiming', 'normal'].")
        
    def forward(self, x: torch.Tensor, expert_mask: torch.Tensor): 
        # CỰC KỲ QUAN TRỌNG: Ép trọng số theo x.dtype ngay lập tức
        A_weight = self.A_stacked.to(dtype=x.dtype)
        B_weight = self.B_stacked.to(dtype=x.dtype)
        
        x_A = torch.einsum('bsi, eir -> bser', x, A_weight) 
        x_AB = torch.einsum('bser, ero -> bseo', x_A, B_weight) 
        
        out = torch.einsum('bseo, bse -> bso', x_AB, expert_mask) 
        return out * self.scaling