import torch 
import torch.nn as nn 
import torch.nn.functional as F
import math 
from core.base_lora import LoRABase

class MoLEExpert(nn.Module): 
    def __init__(self, r: int, lora_alpha: int, hidden_dim: int, init_strategy: str='kaiming'): 
        super().__init__()
        self.rank = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / math.sqrt(r)
        self.init_strategy = init_strategy
        
        self.A = nn.Parameter(torch.empty(r, hidden_dim))
        self.B = nn.Parameter(torch.empty(hidden_dim, r))
        self.reset_parameters()
    
    def reset_parameters(self): 
        if self.init_strategy == 'kaiming': 
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        elif self.init_strategy == 'normal':
            nn.init.normal_(self.A, mean=0.0, std=0.02)
            nn.init.zeros_(self.B)
        else: 
            raise ValueError(f"Chiến lược khởi tạo '{self.init_strategy}' không hợp lệ. Vui lòng chọn từ ['kaiming', 'normal'].")
    
    def forward(self, x: torch.Tensor): 
        # x: batch_size, seq_len, hidden_dim
        x_A = F.linear(x, self.A) # batch_size, seq_len, r
        out = F.linear(x_A, self.B) # batch_size, seq_len, hidden_dim
        return out * self.scaling
class MoLETokenExperts(nn.Module): 
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
        # x: batch_size, seq_len, hidden_dim
        x_A = torch.einsum('bsi, eir -> bser', x, self.A_stacked) # batch_size, seq_len, num_experts, r
        x_AB = torch.einsum('bser, ero -> bseo', x_A, self.B_stacked) # batch_size, seq_len, num_experts, out_features
        
        out = torch.einsum('bseo, bse -> bso', x_AB, expert_mask) # batch_size, seq_len, out_features
        return out * self.scaling
        
class MoLETokenRouter(nn.Module): 
    def __init__(self, hidden_dim: int, num_experts: int): 
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, num_experts)
        self.w2 = nn.Linear(num_experts, num_experts)
    
    def forward(self, x: torch.Tensor): 
        return self.w2(F.tanh(self.w1(x))) 

class ContinualMoLELinear(nn.Module): 
    def __init__(self, base_layer: nn.Linear, 
                 r: int, 
                 lora_alpha: int, 
                 num_token_experts: int=4, 
                 top_k: int=2, 
                 init_strategy: str='kaiming'):
        super().__init__()
        self.base_layer = base_layer
        self.hidden_dim = base_layer.in_features
        self.top_k = top_k
        self.init_strategy = init_strategy
        
        self.r = r 
        self.lora_alpha = lora_alpha
        self.base_layer.requires_grad = False
        
        if self.base_layer.bias is not None: 
            self.bias = nn.Parameter(self.base_layer.bias.data, requires_grad=False)
            
        self.token_router = MoLETokenRouter(self.hidden_dim, num_experts=num_token_experts)
        self.token_experts = MoLETokenExperts(r=r, lora_alpha=lora_alpha, 
                                              in_features=self.hidden_dim, 
                                              out_features=self.base_layer.out_features, 
                                              num_experts=num_token_experts, 
                                              init_strategy=self.init_strategy)
        self.shared_expert = MoLEExpert(r=r, lora_alpha=lora_alpha, hidden_dim=self.hidden_dim, init_strategy=self.init_strategy)
        
        self.num_task = 0
        self.task_keys = nn.Parameter(torch.empty(0, self.hidden_dim), requires_grad=False)
        self.task_experts = nn.ModuleList([])
        # Cache for sentence-based routing
        self.cache_best_task_idx = None 
        self.cache_theta_t = None
        self.cache_theta_IE = None
        
        self.old_token_router = None
        self.old_task_keys = None
        
        self.current_x = None
        self.current_router_logits = None
        
    def on_task_start(self): 
        if self.num_task == 0: 
            import copy 
            self.old_token_router = copy.deepcopy(self.token_router).eval()
            for p in self.old_token_router.parameters(): p.requires_grad = False
            self.old_task_keys = self.task_keys.detach().clone()
            
            for expert in self.task_experts:
                for p in expert.parameters(): p.requires_grad = False
        
        new_key = torch.randn(1, self.hidden_dim, device=self.base_layer.weight.device)
        if self.num_task == 0: 
            self.task_keys = nn.Parameter(new_key, requires_grad=True)
        else: 
            self.task_keys = nn.Parameter(torch.cat([self.task_keys.detach(), new_key], dim=0))
        
        self.task_experts.append(MoLEExpert(r=self.r, 
                                            lora_alpha=self.lora_alpha, 
                                            hidden_dim=self.hidden_dim, 
                                            init_strategy=self.init_strategy))
        self.num_task += 1
    
    def clear_cache(self): 
        self.cache_best_task_idx = None 
        self.cache_theta_t = None
        self.cache_theta_IE = None
        
    def forward(self, x: torch.Tensor): 
        batch_size, seq_len, hidden_dim = x.shape 
        
        base_out = F.linear(x, self.base_layer.weight, getattr(self, 'bias', None)) 
        self.current_x = x
        
        router_logits = self.token_router(x)
        self.current_router_logits = router_logits
        
        routing_probs = F.softmax(router_logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        expert_mask = torch.zeros_like(routing_probs).scatter_(-1, top_k_indices, top_k_probs)
        
        token_expert_outputs = self.token_experts(x, expert_mask)
        
        if self.cache_best_task_idx is not None and self.cache_theta_t is not None and self.cache_theta_IE is not None:
            theta_t = self.cache_theta_t
            theta_IE = self.cache_theta_IE
            theta_t_indices = self.cache_best_task_idx
        else: 
            sentence_representation = x.mean(dim=1) # batch_size, hidden_dim 
            cos_sim = F.cosine_similarity(
                sentence_representation.unsqueeze(1), # batch_size, 1, hidden_dim
                self.task_keys.unsqueeze(0), # 1, num_task, hidden_dim
                dim=-1
            ) # batch_size, num_task
            
            task_scores = F.softmax(cos_sim, dim=-1) # batch_size, num_task
            theta_t, theta_t_indices = torch.max(task_scores, dim=-1) # batch_size
            theta_IE = 1 - theta_t # batch_size 
            
            self.cache_best_task_idx = theta_t_indices
            self.cache_theta_t = theta_t
            self.cache_theta_IE = theta_IE
        task_mask = torch.zeros(batch_size, self.num_task, device=x.device).scatter_(-1, theta_t_indices.unsqueeze(-1), theta_t.unsqueeze(-1)) # batch_size, num_task
        
        task_expert_outputs = torch.zeros_like(base_out)
        
        for i, experts in enumerate(self.task_experts): 
            weight_t = task_mask[:, i].view(-1, 1, 1)
            if weight_t.sum() > 0: 
                delta_out = experts(x) * weight_t
                task_expert_outputs += delta_out
                
        shared_expert_output = self.shared_expert(x) * theta_IE.view(-1, 1, 1)
        
        return base_out + token_expert_outputs + task_expert_outputs + shared_expert_output

    def get_routing_loss(self, gamma: float=1.0, delta: float=1.0): 
        if self.old_token_router is None or self.old_task_keys is None: 
            return torch.tensor(0.0, device=self.current_x.device)
        
        student_router_log_probs = F.log_softmax(self.current_router_logits, dim=-1)
        with torch.no_grad(): 
            teacher_router_logits = self.old_token_router(self.current_x) # batch_size, seq_len, num_experts
            teacher_router_probs = F.softmax(teacher_router_logits, dim=-1) 
        
        L_rkd = F.kl_div(student_router_log_probs, teacher_router_probs, reduction='batchmean')
        
        sentence_representation = self.current_x.mean(dim=1) # batch_size, hidden_dim
        
        old_num_task = self.old_task_keys.size(0)
        student_task_keys_old = self.task_keys[:old_num_task] # old_num_task, hidden_dim
        
        student_cos_sim = F.cosine_similarity(
            sentence_representation.unsqueeze(1), # batch_size, 1, hidden_dim 
            student_task_keys_old.unsqueeze(0), # 1, old_num_task, hidden_dim
            dim=-1) # batch_size, old_num_task
        
        student_keys_log_probs = F.log_softmax(student_cos_sim, dim=-1)
        
        with torch.no_grad(): 
            teacher_cos_sim = F.cosine_similarity(
                sentence_representation.unsqueeze(1), # batch_size, 1, hidden_dim 
                self.old_task_keys.unsqueeze(0), # 1, old_num_task, hidden_dim
                dim=-1) # batch_size, old_num_task
            teacher_keys_probs = F.softmax(teacher_cos_sim, dim=-1)
        l_kkd = F.kl_div(student_keys_log_probs, teacher_keys_probs, reduction='batchmean')
        
        total_routing_loss = gamma * L_rkd + delta * l_kkd
        return total_routing_loss

def flush_mole_cache(model: nn.Module): 
    for module in model.modules(): 
        if isinstance(module, ContinualMoLELinear): 
            module.clear_cache()

def get_total_routing_loss(model: nn.Module, gamma: float=1.0, delta: float=1.0): 
    total_loss = 0.0 
    num_mole_layers = 0
    
    for module in model.modules(): 
        if isinstance(module, ContinualMoLELinear):
            total_loss += module.get_routing_loss(gamma=gamma, delta=delta)
            num_mole_layers += 1
    
    if num_mole_layers > 0:
        return total_loss / num_mole_layers
    else: 
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)