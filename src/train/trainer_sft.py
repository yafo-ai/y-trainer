import json
import logging
import sys
from typing import List
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.nn import functional as F
from src.train.config import TrainConfig
from src.train.data_loader import MultiFormatDataLoader
from src.train.trainer_base import TrainerBase
from src.train.utils import dynamic_sigmoid_batch, masked_mean
from src.utils.scheduler import schedule_element_consumption

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SampleDataset(Dataset):
    def __init__(self, data_path, system_prompt,tokenizer,world_size,batch_size,token_size, max_seq_len=128000):
        """
        初始化数据读取器

        Args:
            file_paths: 传入语料的绝对路径。
            system_prompt: 传入系统提示词。没有默认值，必须强制指定。
            tokenizer: 传入分词器。没有默认值，必须强制指定。
            max_seq_len: 最大序列长度，默认为128000。
        """
        self.file_paths = data_path
        self.system_prompt = system_prompt
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.world_size = world_size
        self.batch_size = batch_size
        self.token_size = token_size
        self.data = self._load_sample()

    def _load_sample(self):
        group_token_samples = []

        loader = MultiFormatDataLoader(
            file_paths=self.file_paths,
            system_prompt=self.system_prompt,
            skip_empty=True
        )
        
        sample_data = loader.load_data()
        for data_item in sample_data:
            sample = self._process_sample(data_item)
            token_samples=self._split_sample(sample)
            group_token_samples.extend(token_samples)

        consumers = self.world_size * self.batch_size
        # 对token样本进行调度分配，确保每个消费者获取的样本相对顺序均衡，避免破坏一组token样本的先后顺序，导致效果下降
        # new_group_token_samples = schedule_element_consumption(group_token_samples, consumers) if consumers > 1 else group_token_samples

        datas=group_token_samples
        # for group in new_group_token_samples:
        #     for token_sample in group:
        #         datas.append(token_sample)
        
        len_datas=len(datas)
        # 截断多余的数据，确保数据量可以被消费者数量整除
        excess = len_datas % consumers
        if excess != 0:
            datas = datas[:len_datas - excess] 
        return datas


    def _load_data(self):
        data = []
        try:
            with open(self.file_paths, 'r', encoding='utf-8') as f:
                datas = json.load(f)
                for idx, dialog in enumerate(datas):
                    data_id = dialog.get("id", idx)
                    if isinstance(data_id, str):
                        try:
                            data_id = int(data_id)
                        except (ValueError, TypeError):
                            data_id = idx
                    
                    data_item = {
                        "system": self.system_prompt,
                        "instruction": dialog.get("instruction", ""),
                        "input": dialog.get("input", ""),
                        "output": dialog.get("output", ""),
                        "data_id": data_id,  # 使用处理后的id值
                    }
                    if len(data_item['output']) == 0 or (len(data_item['instruction'])+len(data_item['input']))==0:
                        continue
                    data.append(data_item)
        except json.JSONDecodeError as e:
            print(f"Failed to load data from file {self.file_paths}: {str(e)}")
        except Exception as e:
            print(f"Unhandled exception while loading data from file {self.file_paths}: {str(e)}")
        return data
    
    def _process_sample(self, dialog):

        full_prompt = self.tokenizer.apply_chat_template([{"role": "system", 'content': f"{dialog['system']}"}, {"role": "user", 'content': f"{dialog['instruction']}{dialog['input']}"}, {"role": "assistant", 'content': f"{dialog['output']}"}], tokenize=False, enable_thinking=False)
        input_prompt = self.tokenizer.apply_chat_template([{"role": "system", 'content': f"{dialog['system']}"}, {"role": "user", 'content': f"{dialog['instruction']}{dialog['input']}"}], tokenize=False, enable_thinking=False, add_generation_prompt=True)

        tokenized = self.tokenizer(full_prompt, return_tensors='pt', add_special_tokens=False)
        prompt_tokenized = self.tokenizer(input_prompt, return_tensors='pt', add_special_tokens=False)
        input_ids = tokenized['input_ids'][0]
        prompt_ids = prompt_tokenized['input_ids'][0]
        output_start = len(prompt_ids)
        
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
    
    def _split_sample(self,sample):
        """ 根据token_size拆分样本。"""
        lables = sample['labels']
        train_token_indices = torch.where(lables != -100)[0]
        token_num = train_token_indices.shape[-1] // self.token_size
        if token_num == 0:
            token_num = 1
        to_lable_idx_start = train_token_indices[0]
        token_sample=[]
        for token_ep in range(token_num):
            start_idx=to_lable_idx_start + token_ep*self.token_size
            end_idx= to_lable_idx_start + token_ep*self.token_size + self.token_size
            token_lable=sample['labels'].clone()
            token_lable[:start_idx]=-100
            token_lable[end_idx:]=-100
            token_item={
                "input_ids": sample['input_ids'],
                "labels": token_lable,
                "attention_mask": sample['attention_mask'],
                "data_id": sample['data_id'],
            }
            token_sample.append(token_item)
        return token_sample
    



    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    


class TrainerSFT(TrainerBase):
    """SFT训练器"""
    def __init__(self, config:TrainConfig,checkpoint_callback):
        super().__init__(config)
        self.checkpoint_callback=checkpoint_callback

    def load_dataset(self,tokenizer)->Dataset:
        #多卡训练时，缩小原来token_batch
        token_size = (self._config.token_batch + self._config.world_size - 1) // self._config.world_size
        sample_dataset = SampleDataset(data_path=self._config.data_path, system_prompt=self._config.system_prompt,tokenizer=tokenizer,world_size=self._config.world_size,batch_size=self._config.batch_size,token_size=token_size,max_seq_len= self._config.max_seq_len)
        if self._config.local_rank == 0:
            print('token_length:', len(sample_dataset))
        return sample_dataset
      

    def collate_fn(self, batch:List[dict],tokenizer):
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

        
    
    def train(self,model, optimizer, batcher,teacher_model=None):

        # unified training loop
        for epoch in range(0, self._config.epoch):
            
            model = self.per_epoch_train(model, optimizer, batcher, epoch, teacher_model)

            self.checkpoint_callback(epoch)

        return model
    
    def per_epoch_train(self,model, optimizer, batcher,epoch, teacher_model=None):

        loss_threshold = 3.0
        loss_deadline = 15.0


        device = f'cuda:{self._config.local_rank}'

        is_distilltion = self._config.distillition

        batch_loop = tqdm(batcher, desc=f"Epoch {epoch}", leave=True)

        for step, batch in enumerate(batch_loop, 1):

            try:

                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

                input_ids = batch.pop('input_ids')
                attention_mask = batch.pop('attention_mask')

                if is_distilltion:

                    with torch.no_grad():
                        input_ids_teacher = input_ids.clone().detach().to('cuda:1')
                        attention_mask_teacher = attention_mask.clone().detach().to('cuda:1')
                        teacher_output = teacher_model(
                            input_ids=input_ids_teacher,
                            attention_mask=attention_mask_teacher,
                        )
                        teacher_logits = teacher_output.logits.view(-1,teacher_output.logits.shape[-1]).to(device)

                labels = batch.pop('labels')

                loss_logs = []
                label_copy = labels.clone().detach()
                to_train_token_idx = torch.where(label_copy.view(-1) != -100)[0]

                label_range=(to_train_token_idx[0],to_train_token_idx[-1])

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits_flatten = outputs.logits.view(-1,outputs.logits.shape[-1])[label_range[0]:label_range[1]]

                targets = labels.clone().detach().view(-1)[label_range[0]:label_range[1]]

                # calculate unweighted loss
                loss_unweighted = torch.nn.functional.cross_entropy(
                    logits_flatten, targets, reduction='none'
                )

                log_loss = loss_unweighted.clone().detach()

                if is_distilltion:
                    
                    prediction_softmax = F.log_softmax(logits_flatten, dim=-1)
                    target_source_softmax = F.softmax(teacher_logits[label_range[0]:label_range[1]], dim=-1)
                    origin_loss_coefficient = self._config.coefficient_of_origin_loss
                    
                    backward_loss = (1 - origin_loss_coefficient) * -1 * (target_source_softmax * prediction_softmax).sum(dim=-1) + (origin_loss_coefficient * loss_unweighted)
                    
                else:
                    backward_loss = loss_unweighted

                if self._config.use_nlirg:
                    # mask = (log_loss > loss_threshold) & (log_loss < loss_deadline)
                    with torch.no_grad():

                        weights = dynamic_sigmoid_batch(
                            log_loss, 
                            max_lr=1,
                            loss_threshold=loss_threshold, 
                            loss_deadline=loss_deadline
                        )
                    
                    # calculate weighted loss
                    loss_weighted_unmeand = backward_loss * weights
                
                    masked_loss_weighted = masked_mean(loss_weighted_unmeand)
                    masked_loss_log = masked_mean(log_loss)
                    loss_log = masked_loss_log.item()
                
                else:
                    
                    # mask = log_loss != 0
                    loss_weighted_unmeand = backward_loss

                    masked_loss_weighted = loss_weighted_unmeand.mean()
                    masked_loss_log = log_loss.mean()
                    loss_log = masked_loss_log.item()
                    if loss_log == None: log_loss = 0

                loss_logs.append(loss_log)

                # loss_weighted_unmeand = loss_weighted_unmeand[mask]

                if optimizer != None:

                    masked_loss_weighted.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                
                else:
                    model.backward(masked_loss_weighted)
                    model.step()
                self.process_logger.write_tb_log(epoch, step, sum(loss_logs)/len(loss_logs))
                del loss_log, logits_flatten, loss_unweighted, targets, outputs, backward_loss
                if 'teacher_output' in locals():
                    del teacher_output
                if 'teacher_logits' in locals():
                    del teacher_logits
                mean_loss = sum(loss_logs)/len(loss_logs)

                
                    
                batch_loop.set_postfix({
                    'progress': f"{batch_loop.n}/{batch_loop.total}",
                    'avg_loss': f"{mean_loss:.4f}",
                })

            except Exception as e:
                logger.error(f"\nError in step {step}: {e}")
                raise

        return model


        

# if __name__ == "__main__":
#     from transformers import AutoModelForCausalLM, AutoTokenizer
#     tokenizer = AutoTokenizer.from_pretrained("E:\\llm_model\\Qwen2.5-0.5B-Instruct", trust_remote_code=True)
#     tokenizer.padding_side = 'left'
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     dataset = SampleDataset("E:\\y-trainer\\y-trainer\\data\\sft_example.json", "", tokenizer,1,1,10)

    