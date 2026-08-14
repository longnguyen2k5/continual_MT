from dataclasses import dataclass, field
from typing import List

@dataclass 
class ExperimentConfig: 
    # General 
    project_name: str
    model_name: str
    learning_rate: float
    batch_size: int
    max_epochs: int
    max_length: int
    num_sample: int 
    num_test_sample: int 
    micro_batch_size: int = 4
    
    # LoRA config
    lora_method: str = 'olora'  # Options: 'lora', 'olora', 'oliera', 'mole'
    r: int = 16
    lora_alpha: int = 1
    target_modules: List[str] = field(default_factory=lambda: ['q_proj', 'v_proj'])
    init_strategy: str = 'kaiming'  # Options: 'kaiming', 'normal', 'svd', 'pissa'
    
    # Mole config
    num_token_experts: int = 4
    top_k: int = 2
    gamma: float = 1.0
    delta: float = 1.0 
    
    # Orthogonal loss weight for LoRA methods
    orthogonal_loss_weight: float = 0.05
    
    # Optional
    fast_dev_run: bool = False
    precision: str = '16-mixed'  # Options: '16-mixed', '32-true', 'bf16-mixed'
    cache_dir: str = './models'
    checkpoint_path: str = './checkpoints'