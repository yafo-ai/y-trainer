import logging
import torch
from torch.utils.data import Dataset
from typing import List
import json

from tqdm import tqdm
from src.train.config import TrainConfig
from src.train.trainer_base import TrainerBase
from src.train.utils import dynamic_sigmoid_batch
# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    

class TrainerCPT(TrainerBase):
    """CPT训练器"""
    def __init__(self, config:TrainConfig):
        super().__init__(config)


    def load_dataset(self,tokenizer)->Dataset:

        cpt_train_texts, data_ids = self._load_texts_from_json()
        packed_input_ids, packed_segment_ids = self._pack_sequences_with_segment_ids(
            cpt_train_texts, 
            data_ids,
            tokenizer=tokenizer
        )
        cpt_dataset = PackedDataset(packed_input_ids, packed_segment_ids)
        
        return cpt_dataset

    def collate_fn(self, batch:List[dict],tokenizer):

        input_ids = [item["input_ids"] for item in batch]
        segment_ids = [item["segment_ids"] for item in batch]
        ids = [item["id"] for item in batch]
        return {
            "input_ids": torch.stack(input_ids),
            "segment_ids": torch.stack(segment_ids),
            "id": ids
        }
        
    
    def train(self,model, optimizer, batcher,teacher_model=None):
        
        for epoch in range(self._config.epoch):

            if self._config.local_rank == 0:
                logger.info(f"Start training Epoch {epoch+1}/{self._config.epoch}.")
            
            batch_loop = tqdm(batcher, desc=f"Epoch {epoch+1}", leave=True)       
            
            for step, batch_cpt in enumerate(batch_loop,1):
                device = f'cuda:{self._config.local_rank}'
                input_ids = batch_cpt["input_ids"].to(device)
                segment_ids = batch_cpt["segment_ids"].to(device)
                sample_id = batch_cpt.get("id", f"cpt_{step}")
                attention_mask_4d = self.prepare_4d_attention_mask(segment_ids, dtype=torch.bfloat16)
                
                weighted_loss, origin_loss = self.compute_weighted_loss(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask_4d,
                    labels=input_ids
                )
                mean_loss = origin_loss.item()

                self.process_logger.write_tb_log(epoch, step, mean_loss)

                if self._config.local_rank == 0:
                    batch_loop.set_postfix({
                        'progress': f"{batch_loop.n}/{batch_loop.total}",
                        'loss':f"{mean_loss:.4f}", 
                    })

                if optimizer == None:
                    model.backward(weighted_loss)
                    model.step()
                else:
                    optimizer.zero_grad()
                    weighted_loss.backward()
                    optimizer.step()
        
                    

        # return model



    def _load_texts_from_json(self) -> List[str]:
        """load texts from json file"""
        texts = []
        ids = []
        with open(self._config.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for num, item in enumerate(data):
                if 'output' in item:
                    texts.append(item['output'])
                    ids.append(item.get('ID', num))
        return texts, ids

    def _pack_sequences_with_segment_ids(self,texts: List[str], data_ids: List, tokenizer) -> tuple:
        """pack segment IDs"""
        packed_input_ids = []
        packed_segment_ids = []
        current_pack = []
        current_segments = []
        current_length = 0
        max_length = self._config.pack_length if self._config.pack_length != -1 else tokenizer.model_max_length
        
        if self._config.pack_length != -1:
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


    def compute_weighted_loss(self,model_engine, input_ids, attention_mask, labels):
        """Compute weighted loss."""

        outputs = model_engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        logits = outputs.logits
        original_loss = outputs.loss

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        if shift_logits.numel() == 0 or shift_labels.numel() == 0:
            return original_loss, original_loss, []
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        loss_per_token_original = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        if self._config.use_nlirg:
            with torch.no_grad():
                weight = dynamic_sigmoid_batch(loss_per_token_original.detach(), max_lr=1, x0=1.2, min_lr=5e-8, k=1.7, loss_threshold=3.0, loss_deadline=15.0)
            loss_per_token = loss_per_token_original * weight
        else:
            loss_per_token = loss_per_token_original

        loss_per_token_original = loss_per_token_original.detach().view(shift_labels.shape)
        loss_per_token = loss_per_token.view(shift_labels.shape)

        
        valid_mask = (shift_labels != -100)
        
        if not self._config.use_nlirg:
            return original_loss, original_loss

        valid_mask = (shift_labels != -100)
        valid_loss = loss_per_token[valid_mask]

        if valid_loss.numel() == 0:
            return original_loss, original_loss

        weighted_loss = valid_loss.mean()
        return weighted_loss, original_loss


    def prepare_4d_attention_mask(self,attention_mask_with_indices, dtype=torch.bfloat16):
        """Generate 4D attention mask from 2D attention mask with indices."""
        bsz, seq_len = attention_mask_with_indices.size()
        min_dtype = torch.finfo(dtype).min
        
        expanded_mask = attention_mask_with_indices[:, None, None, :].expand(bsz, 1, seq_len, seq_len)
        padding_mask = torch.where(expanded_mask != 0, 1, 0).to(dtype=dtype)
        segment_mask = torch.eq(
            expanded_mask, 
            expanded_mask.transpose(-1, -2)
        ).int()
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.int, device=attention_mask_with_indices.device))
        
        combined_mask = segment_mask * causal_mask * padding_mask
        attention_mask_4d = torch.where(
            combined_mask != 0, 
            torch.tensor(0, dtype=dtype, device=attention_mask_with_indices.device), 
            min_dtype
        )
        
        return attention_mask_4d

        