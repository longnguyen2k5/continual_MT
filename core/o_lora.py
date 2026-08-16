import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from core.base_lora import LoRABase

class ContinualLoRABase(LoRABase): 
    def __init__(self, base_layer: nn.Linear, r: int=16, lora_alpha: int=1, init_strategy: str='kaiming'): 
        super().__init__(base_layer, r=r, lora_alpha=lora_alpha, init_strategy=init_strategy)
        self.history_A = nn.ParameterList()
        self.history_B = nn.ParameterList()
        
        self.register_buffer(
            'cache_history_delta_w', 
            torch.zeros(self.out_features, self.in_features),
            persistent=False)
    
    def get_whitelist_keys(self) -> list: 
        return [
            'A_curr',
            'B_curr',
            'history_A',
            'history_B',
            'A_core',
            'B_core',
            'num_reset'
        ]
    
    def prepare_for_loading(self, adapter_state: dict) -> None:
        hist_A_keys = [k for k in adapter_state.keys() if "history_A" in k]
        num_history = len(hist_A_keys)
        
        while len(self.history_A) < num_history:
            idx = len(self.history_A)
            shape_A = adapter_state[f"history_A.{idx}"].shape
            shape_B = adapter_state[f"history_B.{idx}"].shape
            self.history_A.append(nn.Parameter(torch.empty(shape_A), requires_grad=False))
            self.history_B.append(nn.Parameter(torch.empty(shape_B), requires_grad=False))
        
        self.pre_load_undo()
    
    def post_loading_hook(self): 
        self.rebuild_cache()
        self.post_load_redo()
        
    def rebuild_cache(self): 
        with torch.no_grad(): 
            self.cache_history_delta_w.zero_()
            for A_old, B_old in zip(self.history_A, self.history_B): 
                self.cache_history_delta_w += torch.mm(B_old, A_old) # out_features * in_features
                
    def on_task_end(self): 
        with torch.no_grad(): 
            # A_curr: r * in_features
            # B_curr: out_features * r
            if self.init_strategy == 'pissa':
                B_old = torch.concat([self.B_curr.data.clone(), -self.B_core.data.clone()], dim=1) # out_features * (r + r_core)
                A_old = torch.concat([self.A_curr.data.clone(), self.A_core.data.clone()], dim=0) # (r + r_core) * in_features
            else: 
                B_old = self.B_curr.data.clone() # out_features * r
                A_old = self.A_curr.data.clone() # r * in_features
            self.history_A.append(A_old)
            self.history_B.append(B_old)
            self.cache_history_delta_w += torch.mm(B_old, A_old) # out_features * in_features
            self.reset_parameters()
        
    def get_orthogonal_loss(self): 
        loss = 0.0
        num_tasks = len(self.history_B)
        if len(self.history_B) == 0:
            return loss 
        if self.init_strategy == 'pissa':
            A_curr = torch.concat([self.A_curr, self.A_core], dim=0) # (r + r_core) * in_features
            B_curr = torch.concat([self.B_curr, -self.B_core], dim=1) # out_features * (r + r_core)
            for A_old, B_old in zip(self.history_A, self.history_B):
                # shape : B out_features * r_old, A r_old * in_features
                
                B_term = torch.mm(B_old.t(), B_curr) # r_old * (r + r_core)
                A_term = torch.mm(A_curr, A_old.t()) # (r + r_core) * r_old
                
                frobenius_inner = torch.trace(torch.mm(B_term, A_term))
                loss += torch.square(frobenius_inner)
        else: 
            for A_old, B_old in zip(self.history_A, self.history_B):
                A_term = torch.mm(self.A_curr, A_old.t()) # r * r_old
                B_term = torch.mm(B_old.t(), self.B_curr) # r_old * r

                loss += torch.mean(torch.square(A_term)) + torch.mean(torch.square(B_term))
        loss = loss / num_tasks
        return loss
    
    def compute_delta_w(self): 
        delta_w = torch.mm(self.B_curr, self.A_curr)
        total_delta_w = (delta_w + self.cache_history_delta_w) * self.scaling
        return total_delta_w.to(self.target_dtype)
    
class OLoRALinear(ContinualLoRABase): 
    def __init__(self, base_layer: nn.Linear , r: int=16, lora_alpha: int=1, init_strategy: str='pissa'): 
        super().__init__(base_layer, r=r, lora_alpha=lora_alpha, init_strategy=init_strategy)
    def forward(self, x: torch.Tensor):
        delta_w = self.compute_delta_w() # out_features * in_features
        w_active = self.weight + delta_w # out_features * in_features
        return F.linear(x, w_active, self.bias)
    
class OLieRaLinear(ContinualLoRABase): 
    def __init__(self, base_layer: nn.Linear , r: int=16, lora_alpha: int=1, init_strategy: str='pissa'): 
        super().__init__(base_layer, r=r, lora_alpha=lora_alpha, init_strategy=init_strategy)
        
    def forward(self, x: torch.Tensor): 
        delta_w = self.compute_delta_w() # out_features * in_features
        
        scale_factor = torch.exp(delta_w)
        w_active = self.weight * scale_factor # out_features * in_features
        return F.linear(x, w_active, self.bias)
    
def inject_continual_lora(model: nn.Module, 
                          method: str = 'oliera', 
                          r: int = 16, 
                          lora_alpha: int = 1, 
                          target_modules: list = ['q_proj', 'v_proj'],
                          init_strategy: str = 'kaiming'): 
    for name, module in model.named_children(): 
        if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
            TargetClass = OLieRaLinear if method == 'oliera' else OLoRALinear
            new_layer = TargetClass(module, r=r, lora_alpha=lora_alpha, init_strategy=init_strategy)
            
            setattr(model, name, new_layer)
        
        else: 
            inject_continual_lora(module, method=method, r=r, lora_alpha=lora_alpha, target_modules=target_modules, init_strategy=init_strategy)
            
    return model

def get_total_orthogonal_loss(model: nn.Module): 
    total_loss = 0.0 
    for module in model.modules(): 
        if isinstance(module, ContinualLoRABase): 
            total_loss += module.get_orthogonal_loss()
            
    return total_loss
