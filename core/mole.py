import torch 
import torch.nn as nn 
import torch.nn.functional as F
import math 
from core.base_lora import LoRABase

class CosinePrototypeRouter(nn.Module): 
    def __init__(self, in_features: int, temperature: float = 0.1):
        super().__init__()
        
        self.in_features = in_features
        self.temperature = temperature
        
        self.prototype = nn.Parameter(torch.empty(1, self.in_features)) # 1, in_features
        nn.init.normal_(self.prototype, mean=0.0, std=0.02)
    
    def add_task(self): 
        with torch.no_grad(): 
            old_prototype = self.prototype.data.clone()
            num_old_tasks = self.prototype.shape[0]
            
            new_prototype = torch.empty(
                num_old_tasks + 1, 
                self.in_features,
                dtype=old_prototype.dtype
            )
            
            new_prototype[:num_old_tasks] = old_prototype
            nn.init.normal_(new_prototype[num_old_tasks:], mean=0.0, std=0.02)
            self.prototype = nn.Parameter(new_prototype)
        
    def forward(self, x: torch.Tensor):
        # x: batch_size, seq_len, in_features
        pooled_x = x.mean(dim=1) # batch_size, in_features
        
        x_norm = F.normalize(pooled_x, p=2, dim=-1) # batch_size, in_features
        prototype_norm = F.normalize(self.prototype, p=2, dim=-1) # num_tasks, in_features
        
        similarity = torch.matmul(x_norm, prototype_norm.t()) # batch_size, num_tasks
        
        logits = similarity / self.temperature # batch_size, num_tasks
        weight = F.softmax(logits, dim=-1) # batch_size, num_tasks
        return weight.unsqueeze(1) # batch_size, 1, num_tasks
    
    
class MoLELayer(LoRABase): 
    def __init__(self, 
                 base_layer: nn.Linear, 
                 r: int=16, 
                 lora_alpha: int=1, 
                 init_strategy: str='pissa', 
                 temperature: float=0.1):
        super().__init__(base_layer, r=r, lora_alpha=lora_alpha, init_strategy=init_strategy)
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.router = CosinePrototypeRouter(in_features=self.in_features, temperature=temperature)
        self.history_A = nn.ParameterList()
        self.history_B = nn.ParameterList()
    
    def add_task(self): 
        self.router.add_task()
        self.A_curr.requires_grad = False
        self.B_curr.requires_grad = False
        
        A_old = self.A_curr.data.clone()
        B_old = self.B_curr.data.clone()
        
        self.history_A.append(nn.Parameter(A_old, requires_grad=False))
        self.history_B.append(nn.Parameter(B_old, requires_grad=False))
        
        with torch.no_grad(): 
            self.reset_parameters()
        
        self.A_curr.requires_grad = True
        self.B_curr.requires_grad = True
        