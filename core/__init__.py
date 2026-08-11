from .base_lora import StandardLoRALinear
from .o_lora import OLoRALinear, OLieRaLinear
from .mole import ContinualMoLELinear
import torch.nn as nn
import inspect

LORA_MAPPING = {
    'lora': StandardLoRALinear, 
    'olora': OLoRALinear, 
    'oliera': OLieRaLinear, 
    'mole': ContinualMoLELinear
}
def inject_lora(model: nn.Module, 
                method: str = 'oliera', 
                r: int = 16, 
                lora_alpha: int = 1, 
                target_modules: list = ['q_proj', 'v_proj'],
                init_strategy: str = 'kaiming',
                **kwargs): 
    method = method.lower()
    if method not in LORA_MAPPING: 
        raise ValueError(f"Phương pháp '{method}' không hợp lệ. Vui lòng chọn từ {list(LORA_MAPPING.keys())}.")
    
    TargetClass = LORA_MAPPING[method]
    
    sig = inspect.signature(TargetClass.__init__)
    valid_params = sig.parameters.keys() 
    
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    
    for name, module in model.named_children(): 
        if isinstance(module, nn.Linear) and any(target in name for target in target_modules): 
            new_layer = TargetClass(module, r=r, lora_alpha=lora_alpha, init_strategy=init_strategy, **filtered_kwargs)
            setattr(model, name, new_layer)
        else: 
            inject_lora(module, method=method, r=r, lora_alpha=lora_alpha, target_modules=target_modules, init_strategy=init_strategy, **filtered_kwargs)
    
    return model