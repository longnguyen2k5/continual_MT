from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, get_cosine_schedule_with_warmup
import torch
import pytorch_lightning as pl
from core.o_lora import inject_continual_lora, get_total_orthogonal_loss, ContinualLoRABase
import os
import sacrebleu
import torch.nn as nn


class NormalMTModel(pl.LightningModule): 
    def __init__(self, config, tokenizer): 
        super().__init__()
        self.lr = config.get("learning_rate", 0.0003)
        self.ortho_weight = config.get("orthogonal_loss_weight", 0.1)
        self.tokenizer = tokenizer
        self.current_task_name = 'unk'
        self.vi_token_id = self.tokenizer.convert_tokens_to_ids('vie_Latn')
        self.checkpoint_path = config.get("checkpoint_path", "./checkpoints")
        self.max_length = config.get("max_length", 128)
        base_model = self._build_base_model(config)
        self.model = inject_continual_lora(base_model, 
                                           method=config.get('lora_method', 'olora'),
                                           r=config['lora_rank'], 
                                           lora_alpha=config['lora_alpha'],
                                           target_modules=config.get('target_modules', ['q_proj', 'v_proj']),
                                           init_strategy=config.get('init_strategy', 'kaiming')
                                           )
        
        self.print_trainable_parameters()
        
    def _build_base_model(self, config): 
        dtype_map = {
            '16-mixed': torch.float16,
            '32-true': torch.float32,
            'bf16-mixed': torch.bfloat16
        }
        dtype = dtype_map.get(config.get('precision', '16-mixed'), torch.float16)
        
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            config['model_name'], 
            cache_dir=config['cache_dir'],
            use_safetensors=True,
            torch_dtype=dtype
        )
        
        base_model.gradient_checkpointing_enable()
        for param in base_model.parameters():
            param.requires_grad = False
        linear_layers = set()
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Linear):
                # Lấy phần đuôi của tên (ví dụ: 'q_proj' từ 'encoder.layers.0.self_attn.q_proj')
                layer_name = name.split('.')[-1] 
                linear_layers.add(layer_name)

        print("Tên các loại lớp Linear bạn có thể dùng làm target_modules:")
        print(list(linear_layers))
        return base_model
              
    # ===========================
    # 
    # ===========================  
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
        
        total_loss = loss + self.ortho_weight * loss_ortho  
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return total_loss
    
    def on_train_end(self):
        """Hàm này chỉ chạy 1 lần duy nhất khi kết thúc toàn bộ max_epochs"""
        print(f"\n🏁 Đã kết thúc huấn luyện cho task '{self.current_task_name}'. Đang lưu Final Checkpoint...")
        save_dir = self.checkpoint_path
        self.save_smart_checkpoint(save_dir, self.current_task_name)
        
    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * 0.05)  # 5% warmup
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
    def on_validation_epoch_start(self): 
        self.val_preds = [] 
        self.val_refs = []
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx): 
        generated_tokens = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=self.max_length,
            forced_bos_token_id=self.vi_token_id
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
        
        self.val_preds.clear()
        self.val_refs.clear()
        
    def on_test_epoch_start(self): 
        self.on_validation_epoch_start()  # Sử dụng lại logic của validation để chuẩn bị cho test
        
    def test_step(self, batch, batch_idx): 
        self.validation_step(batch, batch_idx)  # Sử dụng lại logic của validation để test
        
    def on_test_epoch_end(self): 
        if not self.val_preds or not self.val_refs:
            return
        
        bleu = sacrebleu.corpus_bleu(self.val_preds, [self.val_refs]).score
        print(f"\n🔥 KẾT QUẢ ĐÁNH GIÁ CHÍNH THỨC - BLEU: {bleu:.2f} 🔥\n")
        self.log("test_bleu", bleu)
        
        self.val_preds.clear()
        self.val_refs.clear()
        
    @torch.no_grad()
    def translate_sentence(self, text): 
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_length=self.max_length, forced_bos_token_id=self.vi_token_id)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    def save_smart_checkpoint(self, save_dir, task_name):
        os.makedirs(save_dir, exist_ok=True)
        full_state_dict = self.model.state_dict()
        lora_keys = [
            "A_curr", 
            "B_curr", 
            "A_core", 
            "B_core",
            "num_reset",
            "history_A", 
            "history_B", 
        ]
        lora_state_dict = {
            k: v for k, v in full_state_dict.items() 
            if any(lora_key in k for lora_key in lora_keys)
        }
        save_path = os.path.join(save_dir, f"checkpoint_{task_name}.pt")
        torch.save(lora_state_dict, save_path)
        print(f"💾 Đã lưu trọng số LoRA và Memory History của task '{task_name}' tại: {save_path}")
        
    def load_smart_checkpoint(self, checkpoint_path): 
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"⚠️ Không tìm thấy checkpoint tại: {checkpoint_path}")
        
        lora_state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        
        count = 0
        for name, module in self.model.named_modules(): 
            if isinstance(module, ContinualLoRABase): 
                hist_A_keys = [k for k in lora_state_dict.keys() if f"{name}.history_A" in k]
                num_history = len(hist_A_keys)
                
                while len(module.history_A) < num_history: 
                    idx = len(module.history_A)
                    shape_A = lora_state_dict[f"{name}.history_A.{idx}"].shape
                    shape_B = lora_state_dict[f"{name}.history_B.{idx}"].shape
                    module.history_A.append(nn.Parameter(torch.empty(shape_A), requires_grad=False))
                    module.history_B.append(nn.Parameter(torch.empty(shape_B), requires_grad=False))
                
                module.pre_load_undo()
                count += 1
                    
        missing_keys, unexpected_keys = self.model.load_state_dict(lora_state_dict, strict=False)
        
        for module in self.model.modules(): 
            if isinstance(module, ContinualLoRABase): 
                module.rebuild_cache()
                module.post_load_redo()
                
        print(f"✅ Đã nạp thành công Checkpoint và đồng bộ W_core cho {count} lớp!")