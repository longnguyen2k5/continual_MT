import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F

class ContinualLoRABase(nn.Module): 
    def __init__(self, base_layer: nn.Linear, r: int=16, lora_alpha: int=1): 
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.scaling = lora_alpha / r 
        
        self.weight = nn.Parameter(base_layer.weight.data, requires_grad=False)
        if base_layer.bias is not None: 
            self.bias = nn.Parameter(base_layer.bias.data, requires_grad=False)
        else: 
            self.register_buffer('bias', None)
            
        self.history_A = nn.ParameterList()
        self.history_B = nn.ParameterList()
        
        self.A_curr = nn.Parameter(torch.empty(self.r, self.in_features))
        self.B_curr = nn.Parameter(torch.empty(self.out_features, self.r))
        
        self.register_buffer(
            'cache_history_delta_w', 
            torch.zeros(self.out_features, self.in_features))
        self.reset_parameters()
        
    def reset_parameters(self): 
        nn.init.normal_(self.A_curr, mean=0.0, std=0.02)
        nn.init.zeros_(self.B_curr)
        
    def add_task(self): 
        with torch.no_grad(): 
                current_delta = torch.mm(self.B_curr, self.A_curr) # out_features * in_features
                self.cache_history_delta_w += current_delta
        self.A_curr.requires_grad = False
        self.B_curr.requires_grad = False
        
        A_old = nn.Parameter(self.A_curr.data.clone(), requires_grad=False)
        B_old = nn.Parameter(self.B_curr.data.clone(), requires_grad=False)
            
        self.history_A.append(A_old)
        self.history_B.append(B_old)
        
        with torch.no_grad(): 
            self.reset_parameters()
            
        self.A_curr.requires_grad = True
        self.B_curr.requires_grad = True
        
        
    def get_orthogonal_loss(self): 
        loss = 0.0
        if len(self.history_B) == 0:
            return loss 
        
        for A_old, B_old in zip(self.history_A, self.history_B): 
            M_A = torch.mm(self.A_curr, A_old.T) # r * r
            loss += torch.sum(torch.square(M_A))
            
            M_B = torch.mm(B_old.T, self.B_curr) # r * r
            loss += torch.sum(torch.square(M_B))
            
        return loss
    
    def compute_delta_w(self): 
        delta_w = torch.mm(self.B_curr, self.A_curr) # out_features * in_features
        return (self.cache_history_delta_w + delta_w) * self.scaling
    

class OLoRALinear(ContinualLoRABase): 
    def forward(self, x: torch.Tensor):
        delta_w = self.compute_delta_w() # out_features * in_features
        w_active = self.weight + delta_w # out_features * in_features
        return F.linear(x, w_active, self.bias)
    
class OLieRaLinear(ContinualLoRABase): 
    def forward(self, x: torch.Tensor): 
        delta_w = self.compute_delta_w() # out_features * in_features
        
        scale_factor = torch.exp(delta_w)
        w_active = self.weight * scale_factor # out_features * in_features
        return F.linear(x, w_active, self.bias)
    
def inject_continual_lora(model: nn.Module, method: str = 'oliera', r: int = 16, lora_alpha: int = 1, target_modules: list = ['q_proj', 'v_proj']): 
    for name, module in model.named_children(): 
        if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
            TargetClass = OLieRaLinear if method == 'oliera' else OLoRALinear
            new_layer = TargetClass(module, r=r, lora_alpha=lora_alpha)
            
            setattr(model, name, new_layer)
        
        else: 
            inject_continual_lora(module, method=method, r=r, lora_alpha=lora_alpha, target_modules=target_modules)
            
    return model

def get_total_orthogonal_loss(model: nn.Module): 
    total_loss = 0.0 
    for module in model.modules(): 
        if isinstance(module, ContinualLoRABase): 
            total_loss += module.get_orthogonal_loss()
            
    return total_loss
