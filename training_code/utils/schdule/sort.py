from transformers import AutoTokenizer, AutoModelForCausalLM

import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def is_garbled(text):
    return "<unk>" in text or "�" in text
    
def token_per_decode(token_ids, tokenizer, max_merge=5):
    output_tokens = []
    i = 0
    n = len(token_ids)
    
    while i < n:
        current_token = tokenizer.decode([token_ids[i]],skip_special_tokens=False)
        
        if not is_garbled(current_token):
            output_tokens.append(current_token)
            i += 1
            continue
            
        merge_count = 1
        ok=False
        while merge_count <= max_merge and i + merge_count <= n:

            merge_ids = token_ids[i:i+merge_count]
            merged_token = tokenizer.decode(merge_ids)
            
            if not is_garbled(merged_token):

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

def model_generate(model,tokenizer,user_input):
    model.eval()
    input_text=tokenizer.apply_chat_template([{"role": "system", 'content': 'You are a helpful assistant.'}, {"role": "user", 'content': user_input}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    inputs = tokenizer(input_text, truncation=True, return_tensors='pt')
    prompt_ids = inputs['input_ids']
    with torch.no_grad():
        prompt_response_ids = model.generate(
            **inputs.to(model.device),
            max_new_tokens=256, 
            num_beams=1,
            num_return_sequences=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            output_attentions=False,
            output_hidden_states=False,
            )
    response_ids = prompt_response_ids[:, prompt_ids.size(1):]

    response_texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    torch.cuda.empty_cache()
    return response_texts[0]

def calculate_loss(user_input, ai_output, model, tokenizer):
    
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
    
        input_ids = model_inputs["input_ids"]
        target_ids = input_ids.clone()
        target_ids[:, :ai_output_start] = -100
        outputs = model(**model_inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
    
        loss_per_token = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                                                        shift_labels.view(-1), reduction='none')


        loss_per_token = loss_per_token.view(input_ids.size(0), -1)
    
        loss_per_token = loss_per_token[0, -ai_output_len:-1]
    
        loss_per_token=loss_per_token.float().cpu().detach().numpy()

    del outputs, logits, shift_logits, shift_labels
    torch.cuda.empty_cache()

    return loss_per_token
   

def calculate_entropy(user_input, ai_output, model, tokenizer):

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
    return entropies


def cosine_similarity_manual(a, b):
    """
    手动计算两个向量a和b的余弦相似度
    """
    # 点积
    dot_product = np.dot(a, b)
    # 范数
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    # 余弦相似度
    return dot_product / (norm_a * norm_b)


def similarity_rank(model, tokenizer, datas):

    similarity_rank = []

    for d in tqdm(datas):

        instruction = d['instruction']

        sft_output = d['output']

        entropy = calculate_entropy(instruction, sft_output, model, tokenizer)

        loss = calculate_loss(instruction, sft_output, model, tokenizer)

        mean_entropy = np.mean(entropy)
        mean_loss = np.mean(loss)
        std_entropy = np.std(entropy)
        std_loss = np.std(loss)

        entropy = (entropy - mean_entropy) / std_entropy
        loss = (loss - mean_loss) / std_loss


        cosine_sim = cosine_similarity_manual(entropy, loss)

        similarity_rank.append(cosine_sim)

    sorted_indexes = np.argsort(similarity_rank)

    return sorted_indexes

def filtered_rank(model, tokenizer, datas):

    
    trigger_element_count = []

    for d in tqdm(datas):
        

        instruction = d['instruction']

        sft_output = d['output']

        entropy = calculate_entropy(instruction, sft_output, model, tokenizer)

        loss = calculate_loss(instruction, sft_output, model, tokenizer)

        selected_element = [
            1 if entropy[i] <= 1 and loss[i] > 3 else 0.5 if entropy[i] >= 1 and loss[i] > 3 else 0
            for i in range(len(entropy))
        ]

        summery = 0
        for i in range(len(selected_element)):
            summery += selected_element[i]

        trigger_element_count.append(summery)
    
    sorted_indexes = np.argsort(trigger_element_count)
           
    return sorted_indexes

def arg_parse():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="", help="data path")
    parser.add_argument("--model_path", type=str, default="", help="model path")
    parser.add_argument("--output_path", type=str, default="", help="output path")
    parser.add_argument("--mode", type=str, default="filtered_rank", help="mode", choices=['similarity_rank','filtered_rank'])
    parser.add_argument("--desc", action="store_true", help="Whether to sort desc")
    args = parser.parse_args()
    return args


def main():

    arg_parser = arg_parse()

    with open(arg_parser.data_path, 'r') as f:
        datas = json.load(f)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(arg_parser.model_path).to(torch.bfloat16).cuda()

    tokenizer = AutoTokenizer.from_pretrained(arg_parser.model_path)

    rank_func = similarity_rank if arg_parser.mode == "similarity_rank" else filtered_rank

    sorted_indices = rank_func(model, tokenizer, datas)

    step = 1 if arg_parser.desc else -1

    sorted_data = []
    for idx in sorted_indices[::step]:
        sorted_data.append(datas[idx])

    with open(arg_parser.output_path, 'w') as f:
        json.dump(sorted_data, f, ensure_ascii=False, indent=4)



main()