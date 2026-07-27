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
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(config_path): 
    pl.seed_everything(42, workers=True)
    config = load_config(config_path)
    
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_name'], 
        cache_dir=config['cache_dir'],
        use_safetensors=True
    )
    
    model = NormalMTModel(config, tokenizer)
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model.model,
        padding=True
    )
    
    continual_task = ['medical', 'it', 'general']
    
    for task_idx, domain_name in enumerate(continual_task): 
        print(f"🚀 Bắt đầu huấn luyện cho domain: {domain_name}")
        
        train_data_list = get_domain_data(domain_name, split_type='train', num_sample=config.get("num_sample", None))
        train_dataset = ContinualTranslationDataset(train_data_list, tokenizer_name_or_path=config['model_name'], max_length=config.get("max_length", 128))
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config.get("batch_size", 2), 
            shuffle=True, 
            collate_fn=data_collator,
            num_workers=0
        )
        
        trainer = pl.Trainer(
            fast_dev_run=config.get("fast_dev_run", False),
            max_epochs=config.get("max_epochs", 3),
            accelerator="auto",
            precision='bf16-mixed'
        )   
        
        validate_datalist = get_domain_data(domain_name, split_type='validation', num_sample=config.get("num_sample", None))
        validate_dataset = ContinualTranslationDataset(validate_datalist, tokenizer_name_or_path=config['model_name'], max_length=config.get("max_length", 128))
        validate_dataloader = DataLoader(
            validate_dataset,
            batch_size=config.get("batch_size", 2),
            shuffle=False,
            collate_fn=data_collator,
            num_workers=0
        )
        
        model.current_task_name = domain_name
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validate_dataloader)
        
        if task_idx < len(continual_task) - 1:  # Nếu chưa phải là task cuối cùng
            model.add_new_task()
            
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
    args = parser.parse_args()
    main(args.config)