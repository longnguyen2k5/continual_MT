import torch
import torch.nn as nn 

class ContinualAdapter(nn.Module): 
    def get_whitelist_keys(self) -> list: 
        return []
    
    def get_adapter_state(self) -> dict: 
        state = self.state_dict(keep_vars=True)
        white_list = self.get_whitelist_keys()
        
        safe_state = {} 
        for key, tensor in state.items(): 
            for white_key in white_list: 
                if key == white_key or key.startswith(white_key + "."): 
                    safe_state[key] = tensor
                    break
        return safe_state
    
    def prepare_for_loading(self, adapter_state: dict) -> None: 
        pass 
    
    def post_loading_hook(self): 
        pass
        