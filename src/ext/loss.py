

import torch

from src.ext.model_loader import ModelLoader
from src.ext.utils import token_per_decode


def calculate_loss(user_input, ai_output,model,tokenizer):

    """
    计算给定用户输入和 AI 输出的每个 token 的损失。

    参数:
    user_input (str): 用户提供的输入文本。
    ai_output (str): AI 生成的输出文本。
    model (torch.nn.Module): 用于计算损失的语言模型。
    tokenizer: 用于将文本转换为 token 的分词器。

    返回:
    tuple: 包含两个元素，第一个元素是 AI 输出的 token 列表，第二个元素是每个 token 的损失数组。
    """
     
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

    ai_output_start=len(system_prompt_ids)+len(user_prompt_start_ids)+len(user_input_ids)+len(user_prompt_end_ids)+len(assistant_start_ids)
    ai_output_len=len(ai_output_ids)

    # 准备模型输入
    attention_mask = [1] * len(full_input_ids)
    model_inputs = {
        "input_ids": torch.tensor([full_input_ids], device=model.device),
        "attention_mask": torch.tensor([attention_mask], device=model.device)
    }

    input_ids= model_inputs["input_ids"]
    target_ids =input_ids.clone()
    target_ids[:, :len(target_ids) - len(ai_output_ids)-len(user_prompt_end_ids)-1] = -100  #只留输出部分

    
    # 计算损失
    with torch.no_grad():
        outputs = model(**model_inputs, labels=target_ids)
        logits = outputs.logits

    # 计算每个 token 的损失
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = target_ids[..., 1:].contiguous()

    # 计算交叉熵损失
    loss_per_token = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                                                    shift_labels.view(-1), reduction='none')

    # 重塑损失以匹配输入的形状
    loss_per_token = loss_per_token.view(input_ids.size(0), -1)

    loss_per_token = loss_per_token[0, ai_output_start-1:ai_output_start+ai_output_len-1]

    loss_per_token=loss_per_token.float().cpu().detach().numpy()

    output_tokens=token_per_decode(ai_output_ids,tokenizer)  #[tokenizer.decode(item) for item in ai_output_ids]


    return output_tokens,loss_per_token
   

# 使用示例
if __name__ == "__main__":
    # 加载模型和分词器

    model_loader = ModelLoader("")
    model_loader.switch_model("D:\\Qwen2.5-0.5B-Instruct")
    tokenizer=model_loader.load_tokenizer()
    model = model_loader.load_model()

    model.eval()
    
    # 定义输入
    context = "华为笔记本内存多大？"
    output_text = "16G"
    
    # 计算注意力分数
    output_tokens, output_loss_per_token= calculate_loss(
        context, output_text, model, tokenizer
    )
