import math
import torch
from torch.utils.data import IterableDataset, Dataset
from typing import Iterator, List
import sys
import json

class RawDataStreamer(IterableDataset):
    def __init__(self, data_path, system_prompt):
        """
        初始化数据读取器

        Args:
            file_paths: 传入语料的绝对路径。
            system_prompt: 传入系统提示词。没有默认值，必须强制指定。
        """
        self.file_paths = data_path
        self.system_prompt = system_prompt
        
    def _stream_data(self):
        try:
            with open(self.file_paths, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for dialog in data:
                    data_id = dialog.get("id", -1)
                    if isinstance(data_id, str):
                        try:
                            data_id = int(data_id)
                        except (ValueError, TypeError):
                            print(f"WARN: data id {data_id} is not a valid integer.")
                    
                    data_item = {
                        "system": self.system_prompt,
                        "instruction": dialog.get("instruction", ""),
                        "input": dialog.get("input", ""),
                        "output": dialog.get("output", ""),
                        "data_id": data_id,  # 使用处理后的id值
                    }
                    yield data_item
        except json.JSONDecodeError as e:
            print(f"Failed to load data from file {self.file_paths}: {str(e)}")
        except Exception as e:
            print(f"Unhandled exception while loading data from file {self.file_paths}: {str(e)}")

    def __iter__(self):
        return self._stream_data()

class TokenLevelSplitter(Dataset):
    def __init__(self, base_dataset, tokenizer, max_seq_len=128000):
     
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.data = []
        for dialog in self.base_dataset:
            self.data.append(self._process(dialog))


    def _process_data(self, dialog):

        full_prompt = self.tokenizer.apply_chat_template([{"role": "system", 'content': f"{dialog['system']}"}, {"role": "user", 'content': f"{dialog['instruction']}{dialog['input']}"}, {"role": "assistant", 'content': f"{dialog['output']}"}], tokenize=False, enable_thinking=False)
        input_prompt = self.tokenizer.apply_chat_template([{"role": "system", 'content': f"{dialog['system']}"}, {"role": "user", 'content': f"{dialog['instruction']}{dialog['input']}"}], tokenize=False, enable_thinking=False, add_generation_prompt=True)

        tokenized = self.tokenizer(full_prompt, return_tensors='pt', add_special_tokens=False)
        prompt_tokenized = self.tokenizer(input_prompt, return_tensors='pt', add_special_tokens=False)
        input_ids = tokenized['input_ids'][0]
        prompt_ids = prompt_tokenized['input_ids'][0]
        output_start = len(prompt_ids)
    
        if dialog['output'] == '':
            return [], []

        return input_ids, output_start
    
    def _process(self, dialog):     
        
        # process data
        input_ids, output_start = self._process_data(dialog)
          
        # if output_start is None(not found start position for SFT) 
        # or the whole sequence is longer than the maximum length, 
        # return empty. That is, the exceeding limit of the corpus 
        # is not split and is not involved in training
        if output_start is None or len(input_ids) > self.max_seq_len:
            return []          
        
        labels = torch.full_like(input_ids[:-1],-100)
        labels[output_start-1:] = input_ids[output_start:]
        attention_mask = torch.ones_like(input_ids[:-1])

        sample = {
            "input_ids": input_ids[:-1],
            "labels": labels,
            "attention_mask": attention_mask,
            "data_id": dialog["data_id"],
        }

        return sample

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

def collate_fn_sft(batch: List[dict], tokenizer) -> dict:
    """
    full the batch and make it to be tensor

    Args:
        batch (List[dict]): batch data list.

    Returns:
        output (dict[tensor]): fulled tensor dict.
    """

    has_data_id = "data_id" in batch[0] if batch else False

    max_len = max(len(x["input_ids"]) for x in batch)
    
    padded_batch = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
    }

    if has_data_id:
        data_ids = []

    for item in batch:
        # padding
        pad_len = max_len - len(item["input_ids"])
        padded_input = torch.cat([
            torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long), 
            torch.tensor(item["input_ids"], dtype=torch.long) 
        ])
        padded_labels = torch.cat([
            torch.full((pad_len,), -100, dtype=torch.long), 
            torch.tensor(item["labels"], dtype=torch.long) 
        ])
        attention_mask = torch.cat([                
            torch.zeros(pad_len, dtype=torch.long),  
            torch.ones(len(item["input_ids"]), dtype=torch.long)  
        ])
        
        padded_batch["input_ids"].append(padded_input)
        padded_batch["labels"].append(padded_labels)
        padded_batch["attention_mask"].append(attention_mask)

        if has_data_id:
            data_ids.append(item["data_id"])     

    
    collected = {
        "input_ids": torch.stack(padded_batch["input_ids"]),
        "labels": torch.stack(padded_batch["labels"]),
        "attention_mask": torch.stack(padded_batch["attention_mask"]),
    }
    
    if has_data_id:
        collected["data_ids"] = torch.tensor(data_ids, dtype=torch.long)

    return collected

    
def process_sft_data(tokenizer, training_config, model_config):
    file_loader = RawDataStreamer(training_config.data_path, model_config.system_prompt)
    token_spliter = TokenLevelSplitter(file_loader, tokenizer, training_config.max_seq_len)
    if training_config.local_rank == 0:
        print('token_length:', len(token_spliter))
    return token_spliter
