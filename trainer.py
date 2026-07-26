import yaml
import torch 
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from core.o_lora import inject_continual_lora, get_total_orthogonal_loss, ContinualLoRABase
from datamodules.dataset import ContinualTranslationDataset, get_domain_data
import os 
import sacrebleu
import gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class NormalMTModel(pl.LightningModule): 
    def __init__(self, config, tokenizer): 
        super().__init__()
        self.lr = config["learning_rate"]
        self.tokenizer = tokenizer
        self.best_bleu_score = -1.0
        self.current_task_name = 'unk'
        
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            config['model_name'], 
            cache_dir=config['cache_dir'],
            use_safetensors=True,
            torch_dtype=torch.bfloat16
        )
        
        base_model.gradient_checkpointing_enable()
        
        linear_layers = set()
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Linear):
                # Lấy phần đuôi của tên (ví dụ: 'q_proj' từ 'encoder.layers.0.self_attn.q_proj')
                layer_name = name.split('.')[-1] 
                linear_layers.add(layer_name)

        print("Tên các loại lớp Linear bạn có thể dùng làm target_modules:")
        print(list(linear_layers))
        
        for param in base_model.parameters():
            param.requires_grad = False
        
        self.model = inject_continual_lora(base_model, method='oliera', r=config['lora_rank'], target_modules=["q_proj", "v_proj"])
        
        self.print_trainable_parameters()
        
        
    def add_new_task(self): 
        count = 0
        for name, module in self.model.named_modules(): 
            if isinstance(module, ContinualLoRABase): 
                module.add_task()
                count += 1
        print(f"✅ Đã thêm task mới cho {count} lớp ContinualLoRA.")
        
    def print_trainable_parameters(self):
        """Hàm tự viết để đếm số lượng tham số được phép huấn luyện"""
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(
            f"🎯 Trainable params: {trainable_params:,} || "
            f"All params: {all_param:,} || "
            f"Trainable%: {100 * trainable_params / all_param:.4f}%"
        )
        
    def training_step(self, batch, batch_idx): 
        outputs = self.model(
            input_ids=batch["input_ids"], 
            attention_mask=batch["attention_mask"], 
            labels=batch["labels"]
        )
        
        loss = outputs.loss
        loss_ortho = get_total_orthogonal_loss(self.model)
        
        total_loss = loss + 0.01 * loss_ortho  # Cộng thêm orthogonal loss với trọng số 0.01
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return total_loss
    
    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        return torch.optim.AdamW(trainable_params, lr=self.lr)
    
    def on_validation_epoch_start(self): 
        self.val_preds = [] 
        self.val_refs = []
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx): 
        vi_token_id = self.tokenizer.convert_tokens_to_ids('vie_Latn')
        generated_tokens = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=128,
            forced_bos_token_id=vi_token_id
        )
        labels = batch['labels']
        labels = torch.where(labels != -100, labels, self.tokenizer.pad_token_id)
        
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        self.val_preds.extend(decoded_preds)
        self.val_refs.extend(decoded_labels)
        
        if batch_idx == 0:
            print(f"\n[👀 MẪU DỊCH THỬ] \n  - Đáp án chuẩn: {decoded_labels[0]} \n  - Mô hình dịch: {decoded_preds[0]}")
            
    def on_validation_epoch_end(self): 
        if not self.val_preds or not self.val_refs:
            print("⚠️ Không có dự đoán hoặc tham chiếu để tính BLEU.")
            return
        
        bleu = sacrebleu.corpus_bleu(self.val_preds, [self.val_refs]).score
        self.log("val_bleu", bleu, prog_bar=True, logger=True)
        print(f"\n🏆 [Epoch {self.current_epoch}] ĐIỂM BLEU: {bleu:.2f}\n")
        
        if bleu > self.best_bleu_score:
            self.best_bleu_score = bleu
            save_dir = "./checkpoints"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"best_model_{self.current_task_name}.pt")
            torch.save(self.model.state_dict(), save_path)
            print(f"💾 Đã lưu mô hình tốt nhất với BLEU {bleu:.2f} tại: {save_path}")
            
        self.val_preds.clear()
        self.val_refs.clear()
        
        
    @torch.no_grad()
    def translate_sentence(self, text): 
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_length=128, forced_bos_token_id=self.tokenizer.convert_tokens_to_ids('vie_Latn'))
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    def save_smart_checkpoint(self, save_dir, task_name):
        os.makedirs(save_dir, exist_ok=True)
        trainable_state_dict = {k: v for k, v in self.model.named_parameters() if v.requires_grad}
        save_path = os.path.join(save_dir, f"checkpoint_{task_name}.pt")
        torch.save(trainable_state_dict, save_path)
        print(f"💾 Đã lưu trọng số LoRA của task '{task_name}' tại: {save_path}")
        
def load_config(): 
    is_kaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') != ''
    config_path = "config/kaggle.yaml" if is_kaggle else "config/local.yaml"
    with open(config_path, "r") as f: 
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    
    pl.seed_everything(42, workers=True)
    config = load_config()
    
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