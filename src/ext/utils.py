
import requests
import torch

from src.ext.dynamic_temperature_generator import DynamicTemperatureGenerator


def get_vllm_embedding(text,model,tokenizer):

    # headers = {"Content-Type": "application/json"}
    # payload = {
    #     "input": text,
    #     "model": model
    # }
    
    # try:
    #     response = requests.post(api_url, headers=headers, json=payload)
    #     response.raise_for_status()  # 检查HTTP错误
    #     return response.json()["data"][0]["embedding"]
    # except Exception as e:
    #     print(f"API调用失败: {str(e)}")
    #     return None


    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    embeddings = model.get_input_embeddings()
    input_embeds = embeddings(input_ids)  # shape: [1, seq_len, hidden_size]

    attention_mask = tokenizer(text, return_tensors="pt")["attention_mask"]
    sentence_embedding = torch.sum(input_embeds * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)
    embedding_list = sentence_embedding[0].tolist()

    return embedding_list
    
    

def attention(full_input_ids,model):
    
    attention_mask = [1] * len(full_input_ids)
    model_inputs = {
        "input_ids": torch.tensor([full_input_ids], device=model.device),
        "attention_mask": torch.tensor([attention_mask], device=model.device)
    }
    attentions = []
    def hook_fn(module, input, output):
        # 捕获注意力权重（假设输出包含注意力）
        if isinstance(output, tuple):
            attentions.append(output[1].detach())  # 索引可能因模型而异

    # 找到最后一层注意力模块
    last_attn_layer = model.model.layers[-1].self_attn  
    # 注册钩子
    hook = last_attn_layer.register_forward_hook(hook_fn)
    # 前向传播获取注意力权重
    with torch.no_grad():
        output = model(**model_inputs)
     # 移除钩子
    hook.remove()
    select_attention = attentions[0]
    attention_weights = select_attention[0].sum(dim=0) 
    attention_weights = attention_weights.float().cpu().detach().numpy()
    return attention_weights


def model_generate_dynamic_tem(model,tokenizer,user_input,temperature=0.5):
    
    input_text=tokenizer.apply_chat_template([{"role": "system", 'content': 'You are a helpful assistant.'}, {"role": "user", 'content': user_input}], add_generation_prompt=True, tokenize=False)

    inputs = tokenizer(input_text, truncation=True, return_tensors='pt')
    prompt_ids = inputs['input_ids']
    generator=DynamicTemperatureGenerator(model,tokenizer,temperature)
    prompt_response_ids = generator.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=1000,
    )
    torch.cuda.empty_cache()
    response_ids = prompt_response_ids[:, prompt_ids.size(1):]

    response_texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

    return response_texts[0]

def model_generate(model,tokenizer,user_input,temperature=0.1,top_p=1.0,top_k=0):

    input_text=tokenizer.apply_chat_template([{"role": "system", 'content': 'You are a helpful assistant.'}, {"role": "user", 'content': user_input}], add_generation_prompt=True, tokenize=False)

    inputs = tokenizer(input_text, truncation=True, return_tensors='pt')
    prompt_ids = inputs['input_ids']
    with torch.no_grad():
        prompt_response_ids = model.generate(
            **inputs.to(model.device),
            max_new_tokens=8520, 
            temperature=temperature,
            top_p = top_p,
            top_k = top_k,
            # 显存优化关键参数
            do_sample=True if temperature>0 else False,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,          # 启用KV缓存
            # renormalize_logits=True,  # 数值稳定性
            output_attentions=False,  # 不返回注意力矩阵
            output_hidden_states=False
            )
    torch.cuda.empty_cache()
    response_ids = prompt_response_ids[:, prompt_ids.size(1):]

    response_texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

    return response_texts[0]

# 辅助函数判断是否乱码（示例实现）
def is_garbled(text):
    # 这里可以替换为实际的乱码检测逻辑
    # 例如检查是否包含非预期字符或编码错误
    return "<unk>" in text or "�" in text
    
def token_per_decode(token_ids, tokenizer, max_merge=5):
    output_tokens = []
    i = 0
    n = len(token_ids)
    
    while i < n:
        current_token = tokenizer.decode([token_ids[i]])
        
        if not is_garbled(current_token):
            output_tokens.append(current_token)
            i += 1
            continue
            
        # 尝试合并多个token
        merge_count = 1
        ok=False
        while merge_count <= max_merge and i + merge_count <= n:
            # 获取当前要合并的token范围
            merge_ids = token_ids[i:i+merge_count]
            merged_token = tokenizer.decode(merge_ids)
            
            if not is_garbled(merged_token):
                # 成功解码，添加到结果并跳过已处理的token
                ok=True
                output_tokens.append(merged_token)
                i += merge_count
                break
            output_tokens.append('')
            merge_count += 1
        if not ok:
            i += max_merge
            output_tokens[i-1]='�'
    
    return output_tokens