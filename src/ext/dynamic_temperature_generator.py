from ast import List
import math
from typing import Union
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM



class DynamicTemperatureGenerator:
    def __init__(self, model,tokenizer,entropy_threshold):
        self.model = model
        self.device = model.device
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        self.max_entropy = math.log(self.vocab_size)
    
        self.entropy_threshold = entropy_threshold  # 熵值阈值，默认0.6
    
    def calculate_entropy(self, logits):
        """计算概率分布的熵值（批量处理）"""
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)  # 防止log(0)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy
    
    def calculate_dynamic_temperature(self, entropy):
        """
        根据熵值动态计算温度
        熵值范围: [0, max_entropy]
        温度范围: [0.1, 1.0]
        """
        # normalized_entropy = entropy / self.max_entropy
        # temperature = 0.1 + 0.9 * normalized_entropy
        # return temperature
        normalized_entropy = entropy / self.max_entropy
        # 使用指数函数确保熵越高温度越高，且低熵区变化平缓
        temperature = 0.1 + 0.9 * (1 - torch.exp(-3 * normalized_entropy))
        return temperature.clamp(min=0.1, max=1.0)
    
    def calculate_dynamic_topk(self, entropy):
        """
        根据熵值动态计算top-k值
        熵值范围: [0, max_entropy]
        topk范围: [1, 10] (整数)
        """
        # normalized_entropy = entropy / self.max_entropy
        # topk = 1 + 9 * normalized_entropy
        # topk = torch.round(topk).long()
        # topk = torch.clamp(topk, min=1, max=10)
        # return topk

        normalized_entropy = entropy / self.max_entropy
        # 使用Sigmoid函数确保平滑过渡到上限
        topk = 1 + 10 * torch.sigmoid(3 * (normalized_entropy - 0.5))
        topk = torch.round(topk).long()
        return topk.clamp(min=1, max=10)
    
    
    def generate(self, input_ids, attention_mask, max_new_tokens=1024):

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        batch_size = input_ids.size(0)
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        cur_input_ids = input_ids
        cur_attention_mask = attention_mask
        
        for _ in range(max_new_tokens):
            if not unfinished.any():
                break
                
            # 获取模型输出
            with torch.no_grad():
                outputs = self.model(
                    input_ids=cur_input_ids,
                    attention_mask=cur_attention_mask
                )
            next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            
            # 计算熵值和动态参数
            entropy = self.calculate_entropy(next_token_logits)
            temperature = self.calculate_dynamic_temperature(entropy)
            topk = self.calculate_dynamic_topk(entropy)
            
            # 创建策略选择掩码
            low_entropy_mask = (entropy < self.entropy_threshold) & unfinished
            high_entropy_mask = ~low_entropy_mask & unfinished
            
            # 初始化下一个token容器
            next_tokens = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            
            # 低熵策略：直接选择最大概率token
            if low_entropy_mask.any():
                next_tokens[low_entropy_mask] = torch.argmax(next_token_logits[low_entropy_mask], dim=-1)
            
            # 高熵策略：动态温度和topk采样
            if high_entropy_mask.any():
                scaled_logits = next_token_logits[high_entropy_mask] / temperature[high_entropy_mask].unsqueeze(1)
                
                # 获取高熵样本的最大topk值
                high_entropy_topk = topk[high_entropy_mask]
                max_k = high_entropy_topk.max().item()
                
                # 取所有高熵样本的topk
                topk_values, topk_indices = torch.topk(scaled_logits, k=max_k, dim=-1)
                
                # 创建自适应掩码
                mask = torch.arange(max_k, device=self.device).expand(len(high_entropy_topk), max_k)
                mask = mask < high_entropy_topk.unsqueeze(1)
                
                # 应用掩码并采样
                masked_values = torch.where(mask, topk_values, torch.tensor(float('-inf'), device=self.device))
                probs = F.softmax(masked_values, dim=-1)
                
                # 随机加权采样，概率高的被采样的概率就高。
                # sampled_indices = torch.multinomial(probs, num_samples=1).squeeze(1)

                # 对概率进行指数平滑（0.5是平滑因子，值越小随机性越强）
                smoothed_probs = probs ** 0.3
                smoothed_probs = smoothed_probs / smoothed_probs.sum(dim=-1, keepdim=True)
                sampled_indices = torch.multinomial(smoothed_probs, num_samples=1).squeeze(1)


                next_tokens[high_entropy_mask] = topk_indices.gather(1, sampled_indices.unsqueeze(1)).squeeze(1)
            
            # 更新完成状态
            eos_mask = next_tokens == self.tokenizer.eos_token_id
            unfinished = unfinished & (~eos_mask)
            
            # 更新注意力掩码
            new_attn_mask = torch.ones(batch_size, 1, dtype=torch.long, device=self.device)
            new_attn_mask[~unfinished] = 0
            
            # 更新当前序列
            cur_input_ids = torch.cat([cur_input_ids, next_tokens.unsqueeze(1)], dim=1)
            cur_attention_mask = torch.cat([cur_attention_mask, new_attn_mask], dim=1)
        
        return cur_input_ids
    