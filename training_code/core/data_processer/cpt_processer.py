import torch
from torch.utils.data import Dataset
from typing import List
import json
from transformers import Qwen2ForCausalLM

def load_texts_from_json(file_path: str) -> List[str]:
    """load texts from json file"""
    texts = []
    ids = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for num, item in enumerate(data):
            if 'output' in item:
                texts.append(item['output'])
                ids.append(item.get('ID', num))
    return texts, ids

class PackedDataset(Dataset):
    """"""
    def __init__(self, input_ids, segment_ids, batch_size=1):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.ids = [f"cpt_{i}" for i in range(len(input_ids))]
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "segment_ids": torch.tensor(self.segment_ids[idx], dtype=torch.long),
            "id": self.ids[idx]
        }      


def pack_sequences_with_segment_ids(texts: List[str], data_ids: List, tokenizer, config) -> tuple:
    """pack segment IDs"""
    packed_input_ids = []
    packed_segment_ids = []
    current_pack = []
    current_segments = []
    current_length = 0
    max_length = config.pack_length if config.pack_length != -1 else tokenizer.model_max_length
    
    if config.pack_length != -1:
        tokenized_texts = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens.append(tokenizer.eos_token_id)
            tokenized_texts.append(tokens)

        for tokens, data_id in zip(tokenized_texts, data_ids):
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            
            if current_length + len(tokens) <= max_length:
                current_pack.extend(tokens)
                current_segments.extend([data_id] * len(tokens))
                current_length += len(tokens)
            else:
                if current_pack:
                    padding_len = max_length - len(current_pack)
                    current_pack += [tokenizer.pad_token_id] * padding_len
                    current_segments += [0] * padding_len
                    packed_input_ids.append(current_pack)
                    packed_segment_ids.append(current_segments)
                
                current_pack = tokens
                current_segments = [data_id] * len(tokens)
                current_length = len(tokens)

        if current_pack:
            padding_len = max_length - len(current_pack)
            current_pack += [tokenizer.pad_token_id] * padding_len
            current_segments += [0] * padding_len
            packed_input_ids.append(current_pack)
            packed_segment_ids.append(current_segments)
    
    else:
        for idx, text in enumerate(texts):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens.append(tokenizer.eos_token_id)

            packs = [tokens[i:i+max_length] for i in range(0, len(tokens), max_length)]
            for pack in packs:
                segment_ids = [data_ids[idx]] * len(pack)
                packed_input_ids.append(pack)
                packed_segment_ids.append(segment_ids)

    return packed_input_ids, packed_segment_ids

def collate_fn_cpt(batch):
    """collate func for cpt"""
    input_ids = [item["input_ids"] for item in batch]
    segment_ids = [item["segment_ids"] for item in batch]
    ids = [item["id"] for item in batch]
    return {
        "input_ids": torch.stack(input_ids),
        "segment_ids": torch.stack(segment_ids),
        "id": ids
    }

def process_cpt_data(tokenizer, training_config):
    
    cpt_train_texts, data_ids = load_texts_from_json(training_config.data_path)
    packed_input_ids, packed_segment_ids = pack_sequences_with_segment_ids(
        cpt_train_texts, 
        data_ids,
        config=training_config,
        tokenizer=tokenizer
    )
    cpt_dataset = PackedDataset(packed_input_ids, packed_segment_ids)
    
    return cpt_dataset