

import torch

from src.ext.model_loader import ModelLoader
from src.ext.utils import token_per_decode
import torch.nn.functional as F

def calculate_entropy(user_input, ai_output,model,tokenizer):

    """
    计算给定用户输入和 AI 输出的每个 token 的熵。

    参数:
    user_input (str): 用户提供的输入文本。
    ai_output (str): AI 生成的输出文本。
    model (torch.nn.Module): 用于计算损失的语言模型。
    tokenizer: 用于将文本转换为 token 的分词器。

    返回:
    tuple: 包含两个元素，第一个元素是 AI 输出的 token 列表，第二个元素是每个 token 的熵数组。
    """
    
    full_prompt = tokenizer.apply_chat_template([{"role": "system", 'content': "You are a helpful assistant."}, {"role": "user", 'content': user_input}, {"role": "assistant", 'content': ai_output}], tokenize=False, enable_thinking=False)
    input_prompt = tokenizer.apply_chat_template([{"role": "system", 'content': "You are a helpful assistant."}, {"role": "user", 'content': user_input}], tokenize=False, enable_thinking=False, add_generation_prompt=True)
 
    full_prompt_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
    input_prompt_ids = tokenizer.encode(input_prompt, add_special_tokens=False)
    
    ai_output_start=len(input_prompt_ids)

    ai_output_ids = full_prompt_ids[ai_output_start:]
    ai_output_len = len(ai_output_ids)
    

    with torch.no_grad():
        
        attention_mask = [1] * len(full_prompt_ids)
        model_inputs = {
            "input_ids": torch.tensor([full_prompt_ids], device=model.device),
            "attention_mask": torch.tensor([attention_mask], device=model.device)
        }
        outputs = model(**model_inputs)
        logits = outputs.logits

        output_logits = logits[0, -ai_output_len:-1]  
    
        probs = F.softmax(output_logits, dim=-1)
        log_probs = F.log_softmax(output_logits, dim=-1)
        entropies = -torch.sum(probs * log_probs, dim=-1).to(torch.float32).cpu().numpy()
        output_tokens=token_per_decode(ai_output_ids, tokenizer)  #[tokenizer.decode(item) for item in ai_output_ids]

        del outputs, logits, output_logits, probs, log_probs
        torch.cuda.empty_cache()

    # 去掉最后一个token，因为它是 EOS 结束后的换行
    if len(output_tokens)-1==len(entropies):
        output_tokens=output_tokens[0:-1]

    return output_tokens,entropies
   

# 使用示例
# if __name__ == "__main__":
#     # 加载模型和分词器
#     model_loader = ModelLoader("")
#     model_loader.switch_model("E:\llm_model\Qwen2.5-0.5B-Instruct","")
#     tokenizer=model_loader.load_tokenizer()
#     model = model_loader.load_model()
    
#     a=tokenizer.decode([198])


#     user_input="你好"
#     ai_output="你好"


#     output_tokens,entropies=calculate_entropy(user_input, ai_output,model,tokenizer)
#     print(output_tokens)
#     print(entropies)

