

import torch
from src.ext.model_loader import ModelLoader
from src.ext.utils import attention, token_per_decode
import numpy as np

def calculate_attention_scores(user_input, ai_output, model, tokenizer, layers=-1):

    # 构建完整的输入文本
    system_prompt = "<|im_start|>system\n<|im_end|>"
    user_prompt_start = "<|im_start|>user\n"
    user_prompt_end="<|im_end|>"
    assistant_start = "<|im_start|>assistant\n"
    
    system_prompt_ids=tokenizer.encode(system_prompt, add_special_tokens=False)
    user_prompt_start_ids=tokenizer.encode(user_prompt_start, add_special_tokens=False)
    user_prompt_end_ids=tokenizer.encode(user_prompt_end, add_special_tokens=False)
    assistant_start_ids=tokenizer.encode(assistant_start, add_special_tokens=False)
    user_input_ids=tokenizer.encode(user_input, add_special_tokens=False)
    ai_output_ids=tokenizer.encode(ai_output, add_special_tokens=False)

    full_input_ids =(system_prompt_ids+user_prompt_start_ids+user_input_ids+user_prompt_end_ids+assistant_start_ids+ai_output_ids+user_prompt_end_ids)


    user_input_start=len(system_prompt_ids)+len(user_prompt_start_ids)
    user_input_len=len(user_input_ids)

    ai_output_start=len(system_prompt_ids)+len(user_prompt_start_ids)+len(user_input_ids)+len(user_prompt_end_ids)+len(assistant_start_ids)
    ai_output_len=len(ai_output_ids)
    full_input_text = tokenizer.decode(full_input_ids[user_input_start:user_input_start+user_input_len])
    print("完整的输入文本:", full_input_text)

    full_input_text = tokenizer.decode(full_input_ids[ai_output_start:ai_output_start+ai_output_len])
    print("完整的输出文本:", full_input_text)
    
   
    attention_weights = attention(full_input_ids,model)
    
    # 提取输出部分对输入部分的注意力
    # 输出token位置范围
    output_attention = attention_weights[ai_output_start:ai_output_start+ai_output_len, user_input_start:user_input_start+user_input_len]
    input_tokens=token_per_decode(user_input_ids,tokenizer)  #[tokenizer.decode(item) for item in user_input_ids]
    output_tokens=token_per_decode(ai_output_ids,tokenizer)  #[tokenizer.decode(item) for item in ai_output_ids]
    # # 获取token文本表示
    return output_attention, input_tokens, output_tokens

# 使用示例
if __name__ == "__main__":
    # 加载模型和分词器

    model_loader = ModelLoader("")
    model_loader.switch_model("D:\\Qwen2.5-0.5B-Instruct")
    tokenizer=model_loader.load_tokenizer()
    model = model_loader.load_model()

    model.eval()
    
    # 定义输入
    context = "# 华为 MateBook D14 2024 13代 酷睿版 笔记本-【皓 月 银】MDG-32 (14寸 i5-13420H+16G+1TB) 内存多大？"
    output_text = "帮我"
    
    # 计算注意力分数
    attention_scores, input_tokens, output_tokens = calculate_attention_scores(
        context, output_text, model, tokenizer
    )
    # 打印结果
    print("输入Tokens:", input_tokens)
    print("输出Tokens:", output_tokens)
    print("注意力分数矩阵形状:", attention_scores.shape)