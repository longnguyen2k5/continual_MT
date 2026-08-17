import torch 
import torch.nn as nn 
import torch.nn.functional as F
import math 
from core.base_adapter import ContinualAdapter
import logging
from core.components import StackedLoRAExperts

# Khởi tạo một logger riêng cho file này
logger = logging.getLogger(__name__)
class MoLEExpert(nn.Module): 
    def __init__(self, r: int, lora_alpha: int, in_features: int, out_features: int, init_strategy: str='kaiming'): 
        super().__init__()
        self.rank = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / math.sqrt(r)
        self.init_strategy = init_strategy
        
        self.A = nn.Parameter(torch.empty(r, in_features))
        self.B = nn.Parameter(torch.empty(out_features, r))
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
        # CỰC KỲ QUAN TRỌNG: Ép trọng số theo x.dtype ngay lập tức
        A_weight = self.A.to(dtype=x.dtype)
        B_weight = self.B.to(dtype=x.dtype)
        
        x_A = F.linear(x, A_weight) 
        out = F.linear(x_A, B_weight) 
        return out * self.scaling
    

class MoLETokenRouter(nn.Module): 
    def __init__(self, hidden_dim: int, num_experts: int): 
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, num_experts)
        self.w2 = nn.Linear(num_experts, num_experts)
    
    def forward(self, x: torch.Tensor): 
        orig_dtype = x.dtype
        x_in = x.to(dtype=self.w1.weight.dtype)
        
        out = self.w2(F.tanh(self.w1(x_in)))
        return out.to(dtype=orig_dtype)
class ContinualMoLELinear(ContinualAdapter): 
    def __init__(self, base_layer: nn.Linear, 
                 r: int, 
                 lora_alpha: int, 
                 num_token_experts: int=4, 
                 top_k: int=2, 
                 init_strategy: str='kaiming', 
                 **kwargs):
        super().__init__()
        self.base_layer = base_layer
        self.hidden_dim = base_layer.in_features
        self.top_k = top_k
        self.init_strategy = init_strategy
        self.debug_mode = kwargs.get('debug_mode', False)
        if self.debug_mode: 
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug mode is enabled for ContinualMoLELinear.")
        self.r = r 
        self.lora_alpha = lora_alpha
        self.base_layer.requires_grad = False
        
        if self.base_layer.bias is not None: 
            self.bias = nn.Parameter(self.base_layer.bias.data, requires_grad=False)
            
        self.token_router = MoLETokenRouter(self.hidden_dim, num_experts=num_token_experts)
        self.token_experts = StackedLoRAExperts(r=r, lora_alpha=lora_alpha, 
                                                in_features=self.hidden_dim, 
                                                out_features=self.base_layer.out_features, 
                                                num_experts=num_token_experts, 
                                                init_strategy=self.init_strategy)
        self.shared_expert = MoLEExpert(r=r, lora_alpha=lora_alpha, in_features=self.hidden_dim, out_features=self.base_layer.out_features, init_strategy=self.init_strategy)
        
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
    
    def get_whitelist_keys(self) -> list:
        return [
            'token_router',
            'token_experts',
            'shared_expert',
            'task_keys',
            'task_experts'
        ]
    
    def prepare_for_loading(self, adapter_state: dict) -> None: 
        if 'task_keys' in adapter_state:
            num_saved_tasks = adapter_state['task_keys'].shape[0]
            
            empty_keys = torch.empty(num_saved_tasks, self.hidden_dim, device=self.base_layer.weight.device, dtype=self.base_layer.weight.dtype)
            self.task_keys = nn.Parameter(empty_keys, requires_grad=False)
            
            while len(self.task_experts) < num_saved_tasks:
                new_expert = MoLEExpert(
                    r=self.r, 
                    lora_alpha=self.lora_alpha, 
                    in_features=self.hidden_dim, 
                    out_features=self.base_layer.out_features,
                    init_strategy=self.init_strategy
                )
                
                # Ép kiểu cho an toàn
                target_dtype = self.base_layer.weight.dtype
                target_device = self.base_layer.weight.device
                new_expert.to(dtype=target_dtype, device=target_device)
                
                self.task_experts.append(new_expert)
                self.num_task += 1
                
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
                                            in_features=self.hidden_dim, 
                                            out_features=self.base_layer.out_features,
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
        
        # --- 1. TOKEN ROUTER (Không bị ảnh hưởng bởi lỗi Cache) ---
        router_logits = self.token_router(x)
        self.current_router_logits = router_logits
        routing_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        expert_mask = torch.zeros_like(routing_probs, device=x.device, dtype=x.dtype).scatter_(-1, top_k_indices, top_k_probs.to(dtype=x.dtype))
        token_expert_outputs = self.token_experts(x, expert_mask)
        
        # --- 2. TASK ROUTER & CACHE LOGIC TỐI THƯỢNG ---
        
        # Điều kiện dùng Cache:
        # CHỈ ĐƯỢC DÙNG KHI: 1. Đang không phải Train, 2. Đang sinh từ (seq_len == 1), 3. Cache có tồn tại
        use_cache = (not self.training) and (seq_len == 1) and \
                    (self.cache_best_task_idx is not None and self.cache_theta_t is not None and self.cache_theta_IE is not None)
        
        if use_cache:
            if self.debug_mode:
                logger.debug(f"🟢 [CACHE] Generate. Batch={batch_size}, Seq={seq_len}")
            # Lấy từ Cache để chạy siêu tốc lúc Generate
            theta_t = self.cache_theta_t.to(device=x.device, dtype=x.dtype)
            theta_IE = self.cache_theta_IE.to(device=x.device, dtype=x.dtype)
            theta_t_indices = self.cache_best_task_idx.to(device=x.device)
        else: 
            # Đang Train, HOẶC đang tính Encoder (seq_len > 1), HOẶC chưa có Cache
            sentence_representation = x.mean(dim=1) # batch_size, hidden_dim 
            cos_sim = F.cosine_similarity(
                sentence_representation.unsqueeze(1), # batch_size, 1, hidden_dim
                self.task_keys.to(device=x.device, dtype=x.dtype).unsqueeze(0), # 1, num_task, hidden_dim
                dim=-1
            ) # batch_size, num_task
            
            task_scores = F.softmax(cos_sim, dim=-1) # batch_size, num_task
            theta_t, theta_t_indices = torch.max(task_scores, dim=-1) # batch_size
            theta_IE = 1 - theta_t # batch_size 
            
            # LƯU CACHE (CHỈ LƯU NẾU ĐANG TEST/GENERATE VÀ BƯỚC ĐẦU TIÊN)
            if (not self.training) and (seq_len > 1):
                if self.debug_mode:
                    logger.debug(f"🟡 [ROUTER] Eval/Encoder (Tạo Cache). Batch={batch_size}, Seq={seq_len}")
                self.cache_best_task_idx = theta_t_indices.to(x.device)
                self.cache_theta_t = theta_t.to(x.device)
                self.cache_theta_IE = theta_IE.to(x.device)
            elif self.training:
                if self.debug_mode:
                    logger.debug(f"🔴 [ROUTER] Train (XÓA CACHE). Batch={batch_size}, Seq={seq_len}")
                # Đang train thì XÓA TRẮNG CACHE để tuyệt đối an toàn
                self.cache_best_task_idx = None
                self.cache_theta_t = None
                self.cache_theta_IE = None

        # --- 3. APPLY TASK EXPERTS ---
        curr_indices = theta_t_indices.to(x.device).unsqueeze(-1)
        curr_src = theta_t.to(x.device).unsqueeze(-1)
        
        task_mask = torch.zeros(batch_size, self.num_task, device=x.device, dtype=x.dtype).scatter_(-1, curr_indices, curr_src.to(dtype=x.dtype))
        
        task_expert_outputs = torch.zeros_like(base_out, device=x.device, dtype=x.dtype)
        
        for i, experts in enumerate(self.task_experts): 
            weight_t = task_mask[:, i].view(-1, 1, 1)
            # Thresholding Optimization
            if weight_t.max().item() > 0.01: 
                delta_out = experts(x) * weight_t
                task_expert_outputs += delta_out
                
        # --- 4. APPLY SHARED EXPERT ---
        shared_expert_output = self.shared_expert(x) * theta_IE.to(x.device).view(-1, 1, 1)
        
        return base_out + token_expert_outputs + task_expert_outputs + shared_expert_output     
    
    def get_routing_loss(self, gamma: float=1.0, delta: float=1.0): 
        if self.old_token_router is None or self.old_task_keys is None: 
            return torch.tensor(0.0, device=self.current_x.device, dtype=self.current_x.dtype)
        
        self.old_token_router.to(self.current_x.device)
        student_router_log_probs = F.log_softmax(self.current_router_logits, dim=-1)
        
        with torch.no_grad(): 
            teacher_router_logits = self.old_token_router(self.current_x) # batch_size, seq_len, num_experts
            teacher_router_probs = F.softmax(teacher_router_logits, dim=-1) 
        
        L_rkd = F.kl_div(student_router_log_probs, teacher_router_probs, reduction='batchmean')
        
        sentence_representation = self.current_x.mean(dim=1) # batch_size, hidden_dim
        
        old_num_task = self.old_task_keys.size(0)
        student_task_keys_old = self.task_keys[:old_num_task].to(dtype=self.current_x.dtype) # old_num_task, hidden_dim
        
        student_cos_sim = F.cosine_similarity(
            sentence_representation.unsqueeze(1), # batch_size, 1, hidden_dim 
            student_task_keys_old.unsqueeze(0), # 1, old_num_task, hidden_dim
            dim=-1) # batch_size, old_num_task
        
        student_keys_log_probs = F.log_softmax(student_cos_sim, dim=-1)
        
        with torch.no_grad(): 
            teacher_cos_sim = F.cosine_similarity(
                sentence_representation.unsqueeze(1), # batch_size, 1, hidden_dim 
                self.old_task_keys.to(device=sentence_representation.device, dtype=self.current_x.dtype).unsqueeze(0), # 1, old_num_task, hidden_dim
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
        return torch.tensor(0.0, device=device, dtype=torch.float32)