import argparse

import torch 
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from datamodules.dataset import ContinualTranslationDataset, get_domain_data
from core.model import NormalMTModel
import os 
import gc
from utils.helpers import load_config
from core.config import ExperimentConfig 

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(config_path, method=None, init_strategy=None): 
    pl.seed_everything(42, workers=True)
    raw_config = load_config(config_path)
    if method is not None: 
        raw_config['lora_method'] = method
    if init_strategy is not None:
        raw_config['init_strategy'] = init_strategy
    
    config = ExperimentConfig(**raw_config)
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, 
        cache_dir=config.cache_dir,
        use_safetensors=True
    )
    model = NormalMTModel(config, tokenizer)
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model.model,
        padding=True
    )
    
    continual_task = ['medical', 'news', 'general']
    
    for task_idx, domain_name in enumerate(continual_task): 
        print(f"🚀 Bắt đầu huấn luyện cho domain: {domain_name}")
        
        train_data_list = get_domain_data(domain_name, split_type='train', num_sample=config.num_sample)
        train_dataset = ContinualTranslationDataset(train_data_list, tokenizer_name_or_path=config.model_name, max_length=config.max_length)
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            collate_fn=data_collator,
            num_workers=0
        )
        
        trainer = pl.Trainer(
            fast_dev_run=config.fast_dev_run,
            max_epochs=config.max_epochs,
            accelerator="gpu",
            devices=1,
            precision=config.precision,  # '16-mixed' | '32-true' | 'bf16-mixed'
        )   
        
        validate_datalist = get_domain_data(domain_name, split_type='validation', num_sample=config.num_sample)
        validate_dataset = ContinualTranslationDataset(validate_datalist, 
                                                       tokenizer_name_or_path=config.model_name, 
                                                       max_length=config.max_length)
        validate_dataloader = DataLoader(
            validate_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=0
        )
        
        model.current_task_name = domain_name
        model.on_task_start()
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validate_dataloader)
        
        if task_idx < len(continual_task) - 1:  # Nếu chưa phải là task cuối cùng
            model.on_task_end()
            
        if hasattr(model, 'trainer'):
            model.trainer = None
        
        del train_dataloader 
        del train_dataset
        del validate_dataloader
        del validate_dataset
        del trainer
        gc.collect()
        
        # 5. Ép GPU xả toàn bộ VRAM của Optimizer cũ
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() # Dọn rác giao tiếp liên tiến trình của CUDA
        
        print(f"✅ Đã giải phóng hoàn toàn bộ nhớ của Task {domain_name}!")
        print("-" * 50)
        
        
    # --- TEST DỊCH TỰ DO SAU KHI TRAIN XONG ---
    print("\n" + "="*50)
    print("🎉 ĐÃ HOÀN THÀNH TOÀN BỘ CONTINUAL LEARNING PIPELINE 🎉")
    print("Thử nghiệm dịch một câu bất kỳ với mô hình cuối cùng:")
    test_sentence = "Deep learning architectures have revolutionized the field of computer vision."
    result = model.translate_sentence(test_sentence)
    print(f"Tiếng Anh: {test_sentence}")
    print(f"Tiếng Việt: {result}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continual Machine Translation")
    parser.add_argument("--config", type=str, default="configs/local.yaml", help="Đường dẫn file config")
    parser.add_argument("--method", type=str, default=None, choices=['lora', 'olora', 'oliera'], 
                        help="Phương pháp muốn chạy (ghi đè file config)")
    parser.add_argument("--init-strategy", type=str, default=None, choices=['pissa', 'kaiming', 'normal', 'svd'], 
                        help="Chiến lược khởi tạo trọng số (ghi đè file config)")
    args = parser.parse_args()
    run_sanity_check_save_load(model, test_method_name="mole")
    main(args.config, method=args.method, init_strategy=args.init_strategy)