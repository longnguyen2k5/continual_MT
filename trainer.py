import argparse

import torch 
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from datamodules.dataset import get_tokenized_dataset
from core.model import NormalMTModel
import os 
import gc
from utils.helpers import load_config
from core.config import ExperimentConfig 
from test_save_load import run_sanity_check_save_load, test_mole_cache_leak

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(config_path, run_sanity_check=False): 
    pl.seed_everything(42, workers=True)
    raw_config = load_config(config_path)
    config = ExperimentConfig(**raw_config)
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, 
        cache_dir=config.cache_dir,
        use_safetensors=True
    )
    model = NormalMTModel(config, tokenizer)
    # =========================================================
    # 2. GỌI SANITY CHECK Ở ĐÂY (TRƯỚC KHI TRAIN)
    # =========================================================
    if run_sanity_check:
        print("🛠️ Đang chạy kiểm thử kiến trúc Save/Load...")
        run_sanity_check_save_load(model, test_method_name=raw_config.get('lora_method', 'mole'))
        test_mole_cache_leak(model, tokenizer)
        print("✅ Kiểm thử xong! Chuẩn bị bước vào quá trình Train thật...\n")
    # =========================================================
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model.model,
        padding=True
    )
    
    continual_task = ['medical', 'news', 'general']
    
    for task_idx, domain_name in enumerate(continual_task): 
        print(f"🚀 Bắt đầu huấn luyện cho domain: {domain_name}")
        
        train_dataset = get_tokenized_dataset(domain_name, tokenizer, 
                                              split_type='train', 
                                              max_length=config.max_length, 
                                              num_sample=config.num_sample, 
                                              cache_dir='./data')
        accumulate_steps = max(1, config.batch_size // config.micro_batch_size)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config.micro_batch_size, 
            shuffle=True, 
            collate_fn=data_collator,
            num_workers=0,
            pin_memory=True # Nên có để transfer từ RAM sang VRAM nhanh hơn
        )
        
        trainer = pl.Trainer(
            fast_dev_run=config.fast_dev_run,
            max_epochs=config.max_epochs,
            accelerator="gpu",
            devices=1,
            precision=config.precision,  # '16-mixed' | '32-true' | 'bf16-mixed'
            accumulate_grad_batches=accumulate_steps,
        )   
        
        # validate_dataset = get_tokenized_dataset(domain_name, tokenizer, 
        #                                          split_type='validation', 
        #                                          max_length=config.max_length, 
        #                                          num_sample=config.num_sample, 
        #                                          cache_dir='./data')
        # validate_dataloader = DataLoader(
        #     validate_dataset,
        #     batch_size=config.micro_batch_size,
        #     shuffle=False,
        #     collate_fn=data_collator,
        #     num_workers=0, 
        #     pin_memory=True # Nên có để transfer từ RAM sang VRAM nhanh hơn
        # )
        
        model.current_task_name = domain_name
        model.on_task_start()
        
        torch.cuda.empty_cache()
        gc.collect()
        
        trainer.fit(model, 
                    train_dataloaders=train_dataloader, 
                    # val_dataloaders=validate_dataloader
                    )
        
        if task_idx < len(continual_task) - 1:  # Nếu chưa phải là task cuối cùng
            model.on_task_end()
            
        if hasattr(model, 'trainer'):
            model.trainer = None
        
        del train_dataloader 
        del train_dataset
        # del validate_dataloader
        # del validate_dataset
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
    parser.add_argument("--sanity-check", action="store_true", 
                        help="Bật cờ này để chạy Sanity Check trước khi huấn luyện (dùng cho debug)")
    
    args = parser.parse_args()
    main(args.config,run_sanity_check=args.sanity_check)