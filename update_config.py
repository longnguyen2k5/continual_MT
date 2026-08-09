import yaml 
import argparse 
import os 

def update_yaml_file(file_path, args_dict): 
    if not os.path.exists(file_path): 
        print(f"⚠️ File '{file_path}' không tồn tại. Không thể cập nhật.")
        return
    
    with open(file_path, 'r') as file: 
        config = yaml.safe_load(file) or {} 
        
    updated = False
    
    for key, value in args_dict.items(): 
        if value is not None: 
            config[key] = value
            updated = True
            
    if updated: 
        with open(file_path, 'w') as file: 
            yaml.dump(config, file)
        print(f"✅ Đã cập nhật '{file_path}' với các giá trị mới.")
    
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Cập nhật file YAML config")
    parser.add_argument("--config", type=str, required=True, help="Đường dẫn tới file config YAML")
    parser.add_argument("--lora_method", type=str, choices=['lora', 'olora', 'oliera'], help="Phương pháp LoRA muốn sử dụng")
    parser.add_argument("--init_strategy", type=str, choices=['pissa', 'kaiming', 'normal', 'svd'], help="Chiến lược khởi tạo trọng số")
    parser.add_argument("--lora_alpha", type=int, help="Hệ số alpha của LoRA (nếu áp dụng)")
    args = parser.parse_args()
    
    args_dict = {
        "lora_method": args.lora_method,
        "init_strategy": args.init_strategy,
        "lora_alpha": args.lora_alpha
    }
    
    update_yaml_file(args.config, args_dict)