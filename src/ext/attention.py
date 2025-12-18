

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

# # 使用示例
# if __name__ == "__main__":
#     # 加载模型和分词器

#     model_loader = ModelLoader("")
#     model_loader.switch_model("E:\llm_model\Qwen2.5-0.5B-Instruct","")
#     tokenizer=model_loader.load_tokenizer()
#     model = model_loader.load_model()

#     model.eval()
    
#     # 定义输入
#     user_input = "你好"
#     ai_output = "你好"


#     full_prompt = tokenizer.apply_chat_template([{"role": "system", 'content': "You are a helpful assistant."}, {"role": "user", 'content': user_input}, {"role": "assistant", 'content': ai_output}], tokenize=False, enable_thinking=False)
#     input_prompt = tokenizer.apply_chat_template([{"role": "system", 'content': "You are a helpful assistant."}, {"role": "user", 'content': user_input}], tokenize=False, enable_thinking=False, add_generation_prompt=True)
#     system_prompt = tokenizer.apply_chat_template([{"role": "system", 'content': "You are a helpful assistant."}], tokenize=False, enable_thinking=False,add_generation_prompt=False)




#     full_prompt_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
#     input_prompt_ids = tokenizer.encode(input_prompt, add_special_tokens=False)
#     system_ids = tokenizer.encode(system_prompt, add_special_tokens=False)


#     print("完整的输入文本:", full_prompt_ids)
#     print("输入部分:", input_prompt_ids)
#     print("系统提示:", system_ids)

#     print("----------------------")
#     print(tokenizer.decode(full_prompt_ids))
#     print("----------------------")
#     print(tokenizer.decode(input_prompt_ids))
#     print("----------------------")
#     print(tokenizer.decode(system_ids))
#     print("----------------------")
   
#     # ai_output_start=len(input_prompt_ids)

#     # ai_output_ids = full_prompt_ids[ai_output_start:]
#     # ai_output_len = len(ai_output_ids)

    
    

#     # 步骤 B: 计算不带生成提示符的用户回合长度，以确定 User 的结束位置
#     # 注意：这里 add_generation_prompt=False，这是关键
    
    