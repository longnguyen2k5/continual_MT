import os
import random
import torch 
from torch.utils.data import Dataset
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer
from datasets import DatasetDict

load_dotenv()  # Load biến môi trường từ file .env nếu có

def get_raw_domain_data(domain_name, split_type='train', cache_dir='./data'):
    if domain_name == 'medical': 
        dataset = load_dataset(
            "mteb/VieMedEVBitextMining", 
            split='test', 
            trust_remote_code=True,
            cache_dir=cache_dir, 
            token=os.getenv("HF_TOKEN")
        )
        train_testval = dataset.train_test_split(test_size=0.2, seed=42)
        test_val = train_testval['test'].train_test_split(test_size=0.5, seed=42)
        
        final_dataset = DatasetDict({
            'train': train_testval['train'],
            'validation': test_val['train'],
            'test': test_val['test']
        })
        target_dataset = final_dataset[split_type]
            
        return [{'en' : ex['sentence2'], 'vi': ex['sentence1']} for ex in target_dataset]
        
    elif domain_name == 'news': 
        dataset = load_dataset(
            'mutiyama/alt', 
            'alt-parallel', 
            trust_remote_code=True, 
            cache_dir=cache_dir, 
            token=os.getenv("HF_TOKEN")
        )
        target_dataset = dataset[split_type] 
        return [{'en' : ex['translation']['en'], 'vi': ex['translation']['vi']} for ex in target_dataset]
    
    elif domain_name == 'general': 
        dataset = load_dataset(
            'thainq107/iwslt2015-en-vi',
            cache_dir=cache_dir, 
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN")
        )
        target_dataset = dataset[split_type]
        return [{'en' : ex['en'], 'vi': ex['vi']} for ex in target_dataset]
    else: 
        raise ValueError(f"Domain '{domain_name}' không hợp lệ. Vui lòng chọn từ ['medical', 'it', 'general'].")
    
def filter_and_valid_data(raw_data_list, max_length): 
    clean_data = [] 
    for item in raw_data_list: 
        src_text = item['en']
        tgt_text = item['vi']
        
        if not src_text or not tgt_text or str(src_text).strip() == "" or str(tgt_text).strip() == "":
            continue
        
        if len(str(src_text).split()) > 0.8 * max_length or len(str(tgt_text).split()) > 0.8 * max_length:
            continue
        
        clean_data.append(item)
    return clean_data

def sample_balanced_data(data_list, num_sample): 
    total_clean = len(data_list)
    random.seed(42)
    
    if total_clean <= num_sample: 
        return data_list
    else: 
        return random.sample(data_list, num_sample)

def get_domain_data(domain_name, split_type='train', max_length=128, num_sample=1000, cache_dir='./data'):
    raw_data = get_raw_domain_data(domain_name, split_type=split_type, cache_dir=cache_dir)
    filtered_data = filter_and_valid_data(raw_data, max_length=max_length)
    sampled_data = sample_balanced_data(filtered_data, num_sample=num_sample)
    
    return sampled_data
        
        
class ContinualTranslationDataset(Dataset): 
    def __init__(self, data_list, tokenizer_name_or_path="facebook/nllb-200-distilled-600M", max_length=128): 
        self.data = data_list
        self.max_length = max_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, 
            src_lang='eng_Latn',
            token=os.getenv("HF_TOKEN")
        )
        
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, idx): 
        item = self.data[idx]
        
        src_text = item['en']
        tgt_text = item['vi']
        
        model_inputs = self.tokenizer(
            text=src_text,
            text_target=tgt_text,
            max_length=self.max_length, 
            truncation=True, 
            return_tensors='pt'
        )
        input_ids = model_inputs['input_ids'].squeeze(0)
        attention_mask = model_inputs['attention_mask'].squeeze(0)
        labels = model_inputs['labels'].squeeze(0)
        
        return {
            'input_ids': input_ids, 
            'attention_mask': attention_mask, 
            'labels': labels
        }