import argparse 
import torch 
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

from utils.helpers import load_config
from core.model import NormalMTModel
from datamodules.dataset import ContinualTranslationDataset, get_domain_data
from core.config import ExperimentConfig

def evaluate(config_path, checkpoint_path, method=None, init_strategy=None): 
    
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
    
    if checkpoint_path is not None:
        model.load_smart_checkpoint(checkpoint_path)
        print(f"🎯 Đang đánh giá mô hình ĐÃ HUẤN LUYỆN (từ {checkpoint_path})")
    else:
        print("⚠️ KHÔNG CÓ CHECKPOINT! Đang đánh giá mô hình GỐC (Zero-shot Baseline)...")
    
    model.eval()
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model.model,
        padding=True
    )
    
    
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        precision='bf16-mixed'
    )
    
    domain_to_test = ['medical', 'news', 'general']
    
    for domain_name in domain_to_test:
        test_data_list = get_domain_data(domain_name, split_type='test', num_sample=config.num_sample)
        test_dataset = ContinualTranslationDataset(test_data_list, tokenizer_name_or_path=config.model_name, max_length=config.max_length)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.batch_size * 2,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=0
        )
        
        trainer.test(model=model, dataloaders=test_dataloader)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Continual Machine Translation Model")
    parser.add_argument("--config", type=str, default="config/local.yaml", help="Path to the config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the model checkpoint")
    parser.add_argument("--method", type=str, default=None, choices=['lora', 'olora', 'oliera', 'mole'], 
                        help="Method to evaluate (overrides config file)")
    parser.add_argument("--init-strategy", type=str, default=None, choices=['pissa', 'kaiming', 'normal', 'svd'], 
                        help="Weight initialization strategy (overrides config file)")
    
    args = parser.parse_args()
    
    evaluate(args.config, args.checkpoint, method=args.method, init_strategy=args.init_strategy)