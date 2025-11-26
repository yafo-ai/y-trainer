from pydantic import BaseModel

class AttentionScoresResponse(BaseModel):
    attention_scores: list[list[float]]  # 注意力分数矩阵
    input_tokens: list[str]              # 输入文本的tokens
    output_tokens: list[str]             # 输出文本的tokens
    origin_output_tokens: list[str]             # 原始输出文本
    origin_attention_scores: list[list[float]]  # 原始输出文本的注意力分数矩阵


class LossResponse(BaseModel):
    output_tokens: list[str]   
    loss_per_token: list[float]

    origin_output_tokens: list[str]             # 原始输出文本
    origin_loss_per_token: list[float]


class CompressResponse(BaseModel):

    attention_scores:list[float]

    input_tokens:list[str]

    compress_text:str


class GenerateSamplesResponse(BaseModel):
    samples:list[str]
    advantage_sample:str


class ClustersResponse(BaseModel):
    clusters:list[int]
    data:list[str]
    random_samples:list[str]
    cluster_dict:dict[int,list[str]]



