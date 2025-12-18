

import re

import numpy as np
import torch
from scipy.signal import convolve2d
from src.ext.model_loader import ModelLoader
from src.ext.utils import attention, token_per_decode

def split_sentence(text,spliter):
    sentence_list = re.split(spliter, text)
    sentence_list = list(filter(None, sentence_list)) # filter 
    sentence_list=[i+spliter for i in sentence_list]
    return sentence_list

def select_sentence_index(selected_token_index, sentences_ids,token_scores,min_scroe):

    targets=[]
    state=[0]*len(sentences_ids)
    for i,sentences in enumerate(sentences_ids):
        state[i]=0
        for id in enumerate(sentences):
            targets.append(i)

    
    for selected_token in selected_token_index:
        sentences_index=targets[selected_token]
        if token_scores[selected_token]>min_scroe:
            state[sentences_index]+=1

    selected_sentence_index=[sentences_ids[i] for i in range(len(state)) if state[i]>0]

    return selected_sentence_index



def compress(context:str,question:str,phrase_token_num:int,top_token_num:int,min_scroe:int,score_type:int,model,tokenizer):
    # 构建完整的输入文本
    system_prompt = "<|im_start|>system\n<|im_end|>"
    user_prompt_start = "<|im_start|>user\n"
    user_prompt_end="<|im_end|>"
    assistant_start = "<|im_start|>assistant\n"
    
    system_prompt_ids=tokenizer.encode(system_prompt, add_special_tokens=False)
    user_prompt_start_ids=tokenizer.encode(user_prompt_start, add_special_tokens=False)
    user_prompt_end_ids=tokenizer.encode(user_prompt_end, add_special_tokens=False)
    assistant_start_ids=tokenizer.encode(assistant_start, add_special_tokens=False)
    
    title_prompt="文档原文：\n"
    title_prompt_ids=tokenizer.encode(title_prompt, add_special_tokens=False)

    question=f"用户聊天记录：\n{question}"
   
    sentences=split_sentence(context,'\n')
    sentences_ids=[tokenizer.encode(sentence, add_special_tokens=False) for sentence in sentences]
    context_ids=[]
    for sentence_ids in sentences_ids:
        context_ids.extend(sentence_ids)
    question_ids=tokenizer.encode(question, add_special_tokens=False)

    full_input_ids =(system_prompt_ids+user_prompt_start_ids+title_prompt_ids+context_ids+question_ids+user_prompt_end_ids+assistant_start_ids)

    context_start=len((system_prompt_ids+user_prompt_start_ids+title_prompt_ids))

    question_start=len((system_prompt_ids+user_prompt_start_ids+title_prompt_ids+context_ids))

    attention_weights = attention(full_input_ids,model)

    question2question = np.copy(attention_weights[question_start:question_start+len(question_ids),question_start:question_start+len(question_ids)])    

    question_weights = np.sum(question2question, axis=1, keepdims=True)

    question2context = attention_weights[question_start:question_start+len(question_ids), context_start:context_start+len(context_ids)]

    if phrase_token_num>1:
        kernel = np.ones((1, phrase_token_num))
        question2context = convolve2d(question2context, kernel, mode="same", boundary="fill", fillvalue=0)

    if score_type==2: #分数加权放大
        question2context = question_weights*question2context
        question2context = question_weights*question2context

    token_scores = np.sum(question2context, axis=0) 

    top_id_indexs = np.argsort(token_scores)[-top_token_num:].tolist()

    select_sentences_ids= select_sentence_index(top_id_indexs,sentences_ids,token_scores,min_scroe)

    compress_text=""
    for sentence_ids in select_sentences_ids:
        compress_text+=tokenizer.decode(sentence_ids)

    input_tokens=token_per_decode(context_ids,tokenizer) 

    return compress_text,input_tokens,token_scores



def extract_feature(context:str,phrase_token_num:int,score_threshold:int,model,tokenizer):
    """
    提取问题的特征
    context:str 文档原文
    question:str 问题
    phrase_token_num:int 卷积核大小
    score_threshold:int  topk
    model: 模型
    tokenizer: 分词器
    """
    # 构建完整的输入文本
    system_prompt = "<|im_start|>system\n<|im_end|>"
    user_prompt_start = "<|im_start|>user\n"
    user_prompt_end="<|im_end|>"
    assistant_start = "<|im_start|>assistant\n"
    
    system_prompt_ids=tokenizer.encode(system_prompt, add_special_tokens=False)
    user_prompt_start_ids=tokenizer.encode(user_prompt_start, add_special_tokens=False)
    user_prompt_end_ids=tokenizer.encode(user_prompt_end, add_special_tokens=False)
    assistant_start_ids=tokenizer.encode(assistant_start, add_special_tokens=False)
    
    title_prompt="文档：\n"
    title_prompt_ids=tokenizer.encode(title_prompt, add_special_tokens=False)

    question=f"问题：\n基于文档的内容提取一个最重要的特征，请注意，你只需要输出一个特征，不需要输出其他内容。"
   
    sentences=split_sentence(context,'\n')
    sentences_ids=[tokenizer.encode(sentence, add_special_tokens=False) for sentence in sentences]
    context_ids=[]
    for sentence_ids in sentences_ids:
        context_ids.extend(sentence_ids)
    question_ids=tokenizer.encode(question, add_special_tokens=False)

    full_input_ids =(system_prompt_ids+user_prompt_start_ids+title_prompt_ids+context_ids+question_ids+user_prompt_end_ids+assistant_start_ids)

    context_start=len((system_prompt_ids+user_prompt_start_ids+title_prompt_ids))

    question_start=len((system_prompt_ids+user_prompt_start_ids+title_prompt_ids+context_ids))

    attention_weights = attention(full_input_ids,model)

    question2question = np.copy(attention_weights[question_start:question_start+len(question_ids),question_start:question_start+len(question_ids)])    

    question_weights = np.sum(question2question, axis=1, keepdims=True)

    question2context = attention_weights[question_start:question_start+len(question_ids), context_start:context_start+len(context_ids)]

    if phrase_token_num>1:
        kernel = np.ones((1, phrase_token_num))
        question2context = convolve2d(question2context, kernel, mode="same", boundary="fill", fillvalue=0)

    #分数加权放大
    question2context = question_weights*question2context
    question2context = question_weights*question2context

    token_scores = np.sum(question2context, axis=0) 

    filtered_indices = np.where(token_scores > score_threshold)[0]

    scores_of_filtered_tokens = token_scores[filtered_indices]

    sorted_order_indices = np.argsort(scores_of_filtered_tokens)[::-1]
    # top_id_indexs = np.argsort(token_scores)[-top_token_num:].tolist()
    final_sorted_indices = filtered_indices[sorted_order_indices]

    feature_text=""
    for index in final_sorted_indices:
        feature_text+=tokenizer.decode(context_ids[index])

    return feature_text



# if __name__ == "__main__":
#     model_loader = ModelLoader("")
#     model_loader.switch_model("D:\\Qwen2.5-0.5B-Instruct")
#     tokenizer=model_loader.load_tokenizer()
#     model = model_loader.load_model()

#     model.eval()
    
#     # 定义输入
    

#     context = """爱普生L11058墨仓式彩色喷墨打印机（A3）[产品id:5A2679]用户：这个能自动打双面吗"""


#     # 计算注意力分数
#     feature_text=extract_feature(context,5,200,model, tokenizer)

#     print(feature_text)

