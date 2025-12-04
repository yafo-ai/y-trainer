from typing import List, Optional
from pydantic import BaseModel, Field

class LoadDataRequest(BaseModel):
    data_path: List[str] = []

class ListDataSetRequest(BaseModel):
    data_dir: str = ""

class LoraRequest(BaseModel):
    lora_path: str = ""

class TrainConfigRequest(BaseModel):
    model_path_to_load: str
    data_path: List[str] = []
    output_dir: str
    use_lora: bool = False
    data_type: str = "bfloat16"
    system_prompt: str = "You are a helpful assistant."
    use_deepspeed: bool = False
    deepspeed_cfg_path: str = ""
    gradit_checkpoing: bool = False
    lora_path: str = ""
    lora_rank: int = 16
    lora_alpha: float = 32
    lora_dropout: float = 0.2
    lora_target_modules: Optional[List[str]] = None
    epoch: int = 3
    lr: float = 5e-6
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    seed: int = 42
    use_tensorboard: bool = False
    tensorboard_path: str = ""
    use_process_bar: bool = False
    training_type: str = "sft"
    checkpoint_epoch: Optional[List[int]] = []
    local_rank: int = 0
    world_size: int = 1
    use_nlirg: bool = True
    token_batch: int = 10
    distillition: bool = False
    teacher_model_path: str = ""
    coefficient_of_origin_loss: float = 0.5
    max_seq_len: int = 20520
    pack_length: int = -1
    rl_type: str = "agent"



# 1. 为 evaluate_loss 接口创建 Pydantic 模型
class EvaluateLossRequest(BaseModel):
    """
    评估模型损失的请求体模型
    """
    data_path: str = Field(..., description="数据集文件的路径", example="/path/to/dataset.json")
    model_path: str = Field(..., description="待评估模型的路径", example="/path/to/model")
    # output_path: str = Field(..., description="评估结果输出路径", example="/path/to/output.json")
    method: str = Field(..., description="评估使用的方法", example="similarity_rank")
    sort: bool = Field(True, description="是否对结果进行排序", example=False)


class EvaluateResultRequest(BaseModel):
    output_path: str = Field(..., description="评估结果输出路径", example="/path/to/output.json")

# 2. 为 merge_models 接口创建 Pydantic 模型
class MergeModelsRequest(BaseModel):
    """
    合并模型的请求体模型
    """
    base_model_path: str = Field(..., description="基础模型的路径", example="/path/to/base_model")
    lora_path: str = Field(..., description="LoRA 适配器的路径", example="/path/to/lora_adapter")
    output_path: str = Field(..., description="合并后模型的输出路径", example="/path/to/merged_model")

