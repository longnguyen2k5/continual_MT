from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, get_cosine_schedule_with_warmup
import torch
import pytorch_lightning as pl
from core import inject_lora
from core.o_lora import get_total_orthogonal_loss
from core.base_adapter import ContinualAdapter
import os
import sacrebleu
import torch.nn as nn
from dataclasses import asdict
from core.mole import flush_mole_cache, get_total_routing_loss 
from core.config import ExperimentConfig

class NormalMTModel(pl.LightningModule): 
    def __init__(self, config: ExperimentConfig, tokenizer): 
        super().__init__()
        self.cfg = config
        self.tokenizer = tokenizer
        self.current_task_name = 'unk'
        self.num_task = 0
        self.vi_token_id = self.tokenizer.convert_tokens_to_ids('vie_Latn')
        base_model = self._build_base_model()
        self.model = inject_lora(base_model, 
                                 method=config.lora_method,
                                 **asdict(self.cfg)
                                )
        
        self.print_trainable_parameters()
        
    def _build_base_model(self): 
        dtype_map = {
            '16-mixed': torch.float16,
            '32-true': torch.float32,
            'bf16-mixed': torch.bfloat16
        }
        dtype = dtype_map.get(self.cfg.precision, torch.float16)
        
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.cfg.model_name, 
            cache_dir=self.cfg.cache_dir,
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
    def on_task_start(self): 
        self.num_task += 1
        count = 0
        for module in self.model.modules(): 
            if hasattr(module, 'on_task_start'): 
                module.on_task_start()
                count += 1
                
        if count > 0:
            print(f"🔄 Đã reset {count} lớp LoRA/MoLE cho task '{self.current_task_name}'")
        
                
    def on_task_end(self): 
        for module in self.model.modules(): 
            if hasattr(module, 'on_task_end'): 
                module.on_task_end()
        
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
        
        task_loss = outputs.loss
        if self.cfg.lora_method in ['olora', 'oliera']:
            loss_ortho = get_total_orthogonal_loss(self.model)
            total_loss = task_loss + self.cfg.orthogonal_loss_weight * loss_ortho
        elif self.cfg.lora_method == 'mole': 
            if self.num_task > 1: 
                routing_loss = get_total_routing_loss(self.model, self.cfg.gamma, self.cfg.delta)
                total_loss = (1 - (self.num_task - 1) / (self.num_task)) * task_loss + (self.num_task - 1) / self.num_task * routing_loss
            else: 
                total_loss = task_loss
        else:
            total_loss = task_loss
            
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return total_loss
    
    def on_train_end(self):
        """Hàm này chỉ chạy 1 lần duy nhất khi kết thúc toàn bộ max_epochs"""
        print(f"\n🏁 Đã kết thúc huấn luyện cho task '{self.current_task_name}'. Đang lưu Final Checkpoint...")
        save_dir = self.cfg.checkpoint_path
        self.save_smart_checkpoint(save_dir, self.current_task_name)
        
    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=self.cfg.learning_rate)
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
        self.val_srcs = [] 
        
    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        if self.cfg.lora_method == 'mole':
            flush_mole_cache(self.model)
            
    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
        if self.cfg.lora_method == 'mole':
            flush_mole_cache(self.model)
            
    @torch.no_grad()
    def validation_step(self, batch, batch_idx): 
        generated_tokens = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=self.cfg.max_length,
            forced_bos_token_id=self.vi_token_id
        )
        labels = batch['labels']
        labels = torch.where(labels != -100, labels, self.tokenizer.pad_token_id)
        
        decoded_srcs = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        self.val_srcs.extend(decoded_srcs)
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
        self.val_srcs.clear()
        
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
        self.val_srcs.clear()
        
    @torch.no_grad()
    def translate_sentence(self, text): 
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_length=self.max_length, forced_bos_token_id=self.vi_token_id)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    def save_smart_checkpoint(self, save_dir, task_name):
        os.makedirs(save_dir, exist_ok=True)
        adapter_state_dict = {}
        
        # 1. Quét toàn bộ model, tìm các Adapter (MoLE hoặc OLoRA đều được)
        for name, module in self.model.named_modules(): 
            if isinstance(module, ContinualAdapter):
                # Bảo Adapter tự nôn dữ liệu của nó ra (dựa trên Whitelist của nó)
                local_state = module.get_adapter_state() 
                for k, v in local_state.items():
                    adapter_state_dict[f"{name}.{k}"] = v
                    
        save_path = os.path.join(save_dir, f"checkpoint_{task_name}.pt")
        torch.save(adapter_state_dict, save_path)
        print(f"💾 Đã lưu Adapter State của task '{task_name}' tại: {save_path}")

    def load_smart_checkpoint(self, checkpoint_path): 
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"⚠️ Không tìm thấy checkpoint tại: {checkpoint_path}")
        
        adapter_state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        count = 0
        
        # --- PHASE 1: CHUẨN BỊ BỘ NHỚ ---
        for name, module in self.model.named_modules(): 
            if isinstance(module, ContinualAdapter):
                prefix = f"{name}."
                # Cắt riêng phần dữ liệu thuộc về module này
                local_state = {k[len(prefix):]: v for k, v in adapter_state_dict.items() if k.startswith(prefix)}
                
                if local_state:
                    # Truyền dữ liệu cho module để nó tự xây chỗ chứa (History hoặc Task Experts)
                    module.prepare_for_loading(local_state)
                    count += 1
                    
        # --- PHASE 2: BƠM DỮ LIỆU ---
        missing_keys, unexpected_keys = self.model.load_state_dict(adapter_state_dict, strict=False)
        
        # --- PHASE 3: XỬ LÝ TOÁN HỌC SAU KHI NẠP ---
        for module in self.model.modules(): 
            if isinstance(module, ContinualAdapter):
                # OLoRA sẽ rebuild_cache, còn MoLE sẽ không làm gì cả!
                module.post_loading_hook()
                
        print(f"✅ Đã nạp thành công Checkpoint cho {count} lớp Adapter!")