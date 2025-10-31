from training_code.utils.evaluate_utils import generate
from training_code.utils.mulit_hot_samples import MultiHotSampleCollection, PROMPT_Agent
from training_code.core.data_processer.rl_processer import processer_rl_data_response
from training_code.core.data_processer.sft_processer import collate_fn_sft
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from .sft_trainer import per_epoch_train

from functools import partial

import json
import threading
import torch
from torch.utils.data import DataLoader 

from tqdm import tqdm

file_cache = {}
cache_lock = threading.Lock()

# def grpo_loss(pi_logprob, pi_old_logprob, pi_ref_logprob, advantage, input_len, len_oi):
#     epsilon = 0.2
#     beta = 0.01

#     bs, seq_len = pi_logprob.shape
#     # skip计算采样的每条采样长度
#     len_oi = torch.tensor([len_oi] * group_num, dtype = torch.long)
#     # 设定mask, 仅对response 为 1， 算loss
#     mask = torch.zeros(bs, seq_len)
#     mask[:, input_len:] = 1

#     # GRPO loss
#     ratio = torch.exp(pi_logprob - pi_old_logprob)
#     ratio_clip = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
#     advantage = advantage.unsqueeze(dim = 1) # [a, b ,c] -> [[a], [b], [c]]
#     policy_gradient = torch.minimum(ratio * advantage , ratio_clip * advantage)
#     kl = grpo_kl(pi_logprob, pi_ref_logprob)

#     loss = (policy_gradient -  beta * kl) * mask
#     loss = (-1 / group_num ) * (1/len_oi.unsqueeze(dim = 1)) * loss  
#     loss = loss.sum()

#     return loss

def compare_last_epoch_is_raise(epoch,batch,id,cscore):
    """与之前的epoch比较分数，要保证分数增长，才有强化训练得必要"""
    if epoch==1:
        return True
    while epoch > 1:
        epoch-=1
        try:
            with cache_lock:
                if f'{epoch}_{batch}' not in file_cache:
                    with open(f'data/{epoch}_{batch}_raws_data.json', 'r', encoding='utf-8') as f:
                        samples = json.load(f)
                        file_cache[f'{epoch}_{batch}']=samples
                samples=file_cache[f'{epoch}_{batch}']
            for sample in samples:
                if sample['id'] == id:
                    last_epoch_score = sample['score']
                    return last_epoch_score < cscore
        except:
            continue
    return True

def generate_samples_worker(model, tokenizer, prompt_queue, samples_queue):
    """工作线程函数，用于生成样本"""
    while True:
        print("generate_samples_worker")
        try:
            id,prompt,input = prompt_queue.get(timeout=1)
            print(f"{id}_正在生成")
            samples = generate(model, tokenizer, prompt)
            samples_queue.put((id, input,prompt,samples))
            prompt_queue.task_done()
        except prompt_queue.Empty:
            break
        except Exception as e:
            print(f"generate_samples_worker: {e}")
            prompt_queue.task_done()


def rl_train_model(model, tokenizer, optimizer, batcher, training_config, model_config):
    
    for epoch in range(training_config.epoch):
        batch_loop = tqdm(batcher, desc=f"Epoch {epoch}", leave=True)

        for batch_id, batch in enumerate(batch_loop):
            processed_data = []
            log_data=[]
            model.eval()

            with torch.no_grad():
                for id,input in zip(batch["ids"],batch["inputs"]):

                    sampleCollection=MultiHotSampleCollection(id,input,model,tokenizer)
                    sampleCollection.create_samples_tree(sampleCollection.root)
                    _data,_log=sampleCollection.get_all_samples()
                    processed_data.extend(_data)
                    log_data.extend(_log)

            if processed_data == []:
                continue

            response_dataset = processer_rl_data_response(processed_data, tokenizer, training_config, model_config)
            
            collate_fn = partial(collate_fn_sft, tokenizer=tokenizer)
            if model_config.use_deepspeed:
                sampler = DistributedSampler(batcher, rank=dist.get_rank(), num_replicas=dist.get_world_size(),shuffle=False)
            else:
                sampler = None

            rl_batch = DataLoader(response_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, sampler=sampler)
            
            
            model = per_epoch_train(model, optimizer, rl_batch, training_config, epoch)

        Checkpoint_epoch = training_config.checkpoint_epoch if isinstance(training_config.checkpoint_epoch, list) else [training_config.checkpoint_epoch]

        if Checkpoint_epoch == []:
            continue

        # 在第Checkpoint_epoch轮结束时保存检查点
        if epoch in [e - 1 for e in Checkpoint_epoch]:  # 0-based索引，第三轮对应索引2
            
            # 保存检查点
            current_checkpoint_dir = f"{training_config.output_dir}/checkpoint-{epoch+1}"
            # 使用peft官方方法保存适配器
            model.save_pretrained(
                save_directory=current_checkpoint_dir,
                save_adapter=model_config.use_lora,  # 仅保存LoRA适配器
                safe_serialization=True  # 使用safetensors格式
            )
            # optimizer.save_state_to_file(f"{traing_config.output_dir}/optimizer.pt")
            print(f"\n=== 已到达训练检查点轮数{epoch}，LoRA适配器已保存到 {current_checkpoint_dir} ===")

    return model




            

            

if __name__ == "__main__":
    pass
    # from transformers import AutoTokenizer, AutoModelForCausalLM
    # from peft import PeftModel

    # tokenizer = AutoTokenizer.from_pretrained("/root/autodl-fs/model/Qwen3-8B")
    # # model = AutoModelForCausalLM.from_pretrained("/root/autodl-fs/model/base_model_qwen7b")
    # # model = PeftModel.from_pretrained(model, "/root/autodl-fs/trained_model/lora/v2.72.5")
    # # model.to("cuda")
    # question = '用户正在浏览：【华为原厂配件】华为 MatePad 11.5S 星闪键盘TGR-KB11-【大象灰】[产品id:2095453]\n用户：[图片]\n用户：好的，有放星闪笔的位置？'

    # # prompted_question = PROMPT_Agent.format(input=question)

    # normal_question = '你好,你是谁'
    # full_prompt = tokenizer.apply_chat_template([{"role": "system", 'content': f"{normal_question}"}, {"role": "user", 'content': f"{normal_question}"}, {"role": "assistant", 'content': f"{normal_question}"}], add_generation_prompt=False, tokenize=False, enable_thinking=False)
    # print(full_prompt)
    # input_text=tokenizer.apply_chat_template([{"role": "system", 'content': 'You are a helpful assistant.'}, {"role": "user", 'content': prompted_question}], add_generation_prompt=True, tokenize=False)
    # samples = generate(model, tokenizer, prompted_question)
    # inputs = tokenizer(input_text, return_tensors='pt')
    # prompt_response_ids=model.generate(**inputs.to(model.device), 
    #                                             max_length=1024,
    #                                             # num_beams=1,
    #                                             temperature=0.8,
    #                                             top_p = 0.9,
    #                                             do_sample=True,
    #                                             num_return_sequences=5,
    #                                             pad_token_id=tokenizer.eos_token_id,
    #                                             eos_token_id=tokenizer.eos_token_id, 
    #                                             output_attentions=False,  # 不返回注意力矩阵
    #                                             output_hidden_states=False,
    #                                             )
    # # o = choose_sample(samples,prompted_question)
    # print(prompt_response_ids)
    # for k in tokenizer.batch_decode(prompt_response_ids, skip_special_tokens=False):
    #     print(k.replace(input_text,'').replace('<|im_end|>',''),end="\n")
    #     print('-'*30)
    #     print('\n')

    # sampleCollection=MultiHotSampleCollection(0,question,model,tokenizer)
    # llm_outputs=sampleCollection.llm_generates(question)
    # print(llm_outputs)