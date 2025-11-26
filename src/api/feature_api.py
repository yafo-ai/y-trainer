import random
from fastapi import APIRouter, Body, UploadFile, Form
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import torch

from src.ext.attention import calculate_attention_scores
from src.ext.clusterer import y_cluster
from src.ext.entropy import calculate_entropy
from src.ext.loss import calculate_loss
from src.ext.utils import model_generate, model_generate_dynamic_tem
from src.models.ext_response import AttentionScoresResponse, ClustersResponse, CompressResponse, GenerateSamplesResponse, LossResponse
from src.ext.model_loader import global_model_loader
from src.ext.atten_compress import compress,extract_feature
from src.ext.utils import get_vllm_embedding


router = APIRouter(
    prefix="/api/feature",
    tags=['特征分析']
)

@router.post("/clusters", summary="文本聚类")
def clusters_text(data:list[str] = Body(description="需要聚类的文本列表"),
                 is_attent:bool=Body(description="是否使用注意力提取特征",default=False),
                 score_threshold:int=Body(description="注意了提取多少个token特征"),default=1500):

    tokenizer=global_model_loader.load_tokenizer() if is_attent else None
    model = global_model_loader.load_model() if is_attent else None

    datas=[]
    for text in data:
        if is_attent:
            feature_text=extract_feature(text,5,score_threshold,model,tokenizer)
            if feature_text=="":
                raise ValueError("注意力分数过大，没有提取到相关特征！")

            datas.append(get_vllm_embedding(feature_text,model,tokenizer))
        else:
            datas.append(get_vllm_embedding(text,model,tokenizer))

    clusters=y_cluster(datas)


    # 根据聚类信息将文本分组
    cluster_dict = {}
    for text, cluster in zip(data, clusters):
        if cluster not in cluster_dict:
            cluster_dict[cluster] = []
        cluster_dict[cluster].append(text)

    # 从每个组中随机选择一个样本
    random_samples = [random.choice(cluster_dict[cluster]) for cluster in cluster_dict]
    
    response=ClustersResponse(
        clusters=clusters,
        data=data,
        random_samples=random_samples,
        cluster_dict=cluster_dict
    )

    return response




@router.post("/attention_scores", summary="计算注意力分数")
def attention_scores(input: str = Body(description="输入"),output: str = Body(description="输出")):
  
    
    tokenizer=global_model_loader.load_tokenizer()
    model = global_model_loader.load_model()

    origin_output = model_generate(model,tokenizer,input)

    origin_attention_scores, input_tokens, origin_output_tokens = calculate_attention_scores(input, origin_output, model, tokenizer)

    attention_scores, input_tokens, output_tokens = calculate_attention_scores(input, output, model, tokenizer)
    
    response = AttentionScoresResponse(
        attention_scores=attention_scores,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        origin_output_tokens=origin_output_tokens,
        origin_attention_scores=origin_attention_scores

    )
    return response


@router.post("/loss_scores", summary="计算逐个token损失")
def loss_scores(input: str = Body(description="输入"),output: str = Body(description="输出")):
  
    tokenizer=global_model_loader.load_tokenizer()
    model = global_model_loader.load_model()

    origin_output = model_generate(model,tokenizer,input)

    origin_output_tokens, origin_loss_per_token = calculate_loss(input, origin_output, model, tokenizer)

    output_tokens,loss_per_token = calculate_loss(input, output, model, tokenizer)
    
    response = LossResponse(
       
        output_tokens=output_tokens,
        loss_per_token=loss_per_token,
        origin_output_tokens=origin_output_tokens,
        origin_loss_per_token=origin_loss_per_token

    )
    
    return response

@router.post("/entropy_scores", summary="计算逐个token熵")
def entropy_scores(input: str = Body(description="输入"),output: str = Body(description="输出")):
  
    tokenizer=global_model_loader.load_tokenizer()
    model = global_model_loader.load_model()

    origin_output = model_generate(model,tokenizer,input)

    origin_output_tokens, origin_loss_per_token = calculate_entropy(input, origin_output, model, tokenizer)

    output_tokens,loss_per_token = calculate_entropy(input, output, model, tokenizer)
    
    response = LossResponse(
       
        output_tokens=output_tokens,
        loss_per_token=loss_per_token,
        origin_output_tokens=origin_output_tokens,
        origin_loss_per_token=origin_loss_per_token

    )
    
    return response

@router.post("/compress", summary="压缩文本")
def attention_compress(input: str = Body(description="输入"),question: str = Body(description="问题"),kernel_num:int = Body(description="卷积核"),topk:int = Body(description="topk"),min_scroe:int = Body(description="最小分数线"),score_type:int=Body(description="分数类型1：原始分数，2：分数加权放大",default=1)):
    tokenizer=global_model_loader.load_tokenizer()
    model = global_model_loader.load_model()
    
    compress_text,input_tokens,attention_scores= compress(input, question,kernel_num,topk,min_scroe,score_type, model, tokenizer)

    return  CompressResponse(
        attention_scores=attention_scores,
        input_tokens=input_tokens,
        compress_text=compress_text,
    )

@router.post("/generate_samples", summary="样本生成")
def generate_samples(input: str = Body(description="输入"),prompt_temp:str=Body(description="提示词模板"),generate_num:int = Body(description="生成样本数",default=5),temperature:float=Body(description="温度",default=0.1),top_p:float = Body(description="top_p",default=1.0),top_k:int = Body(description="top_k",default=0)):

    
    tokenizer=global_model_loader.load_tokenizer()
    model = global_model_loader.load_model()
    
    samples=[]
    samples_text=""
    for i in range(generate_num):
        origin_output = model_generate(model,tokenizer,input,temperature,top_p,top_k)
        samples.append(origin_output)
        samples_text+=(origin_output+"\n\n")

    #反思，挑选优势样本并修复样本内的问题
    # prompt_temp=f"# 你是一个样本挑选助手，我会提供任务背景与在这个任务背景下生成的多组样本，你需要从这些样本中挑选出针对任务背景最有优势的一个样本，并修复样本内的问题并加强样本的优势。\n\n ## 任务背景 \n\n {input} \n\n ## 多组样本 \n\n {samples} \n\n ## 注意只需要你输出优势样本，不要输出无关内容，\n最后挑选的一个样本是："
    input_samples=prompt_temp.format(input=input, samples=samples_text)
    advantage_sample = model_generate(model,tokenizer,input_samples)

    return GenerateSamplesResponse(samples=samples,advantage_sample=advantage_sample)

@router.post("/dynamic_generate_samples", summary="样本生成")
def dynamic_generate_samples(input: str = Body(description="输入"),prompt_temp:str=Body(description="提示词模板"),generate_num:int = Body(description="生成样本数",default=5),temperature:float=Body(description="温度",default=0.1),top_p:float = Body(description="top_p",default=1.0),top_k:int = Body(description="top_k",default=0)):

    
    tokenizer=global_model_loader.load_tokenizer()
    model = global_model_loader.load_model()
    
    samples=[]
    samples_text=""
    for i in range(generate_num):
        origin_output = model_generate_dynamic_tem(model,tokenizer,input,temperature)
        samples.append(origin_output)
        samples_text+=(origin_output+"\n\n")

    #反思，挑选优势样本并修复样本内的问题
    # prompt_temp=f"# 你是一个样本挑选助手，我会提供任务背景与在这个任务背景下生成的多组样本，你需要从这些样本中挑选出针对任务背景最有优势的一个样本，并修复样本内的问题并加强样本的优势。\n\n ## 任务背景 \n\n {input} \n\n ## 多组样本 \n\n {samples} \n\n ## 注意只需要你输出优势样本，不要输出无关内容，\n最后挑选的一个样本是："
    input_samples=prompt_temp.format(input=input, samples=samples_text)
    advantage_sample = model_generate_dynamic_tem(model,tokenizer,input_samples,temperature)

    return GenerateSamplesResponse(samples=samples,advantage_sample=advantage_sample)