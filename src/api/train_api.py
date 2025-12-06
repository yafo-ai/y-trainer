import asyncio
import contextlib
import json
import logging
import os
import sys
import time
from fastapi import APIRouter, Body, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
import torch
from tqdm import tqdm
from datetime import datetime
from src.evaluate.entropy_sort import evaluate_loss
from src.models.train_model import EvaluateLossRequest, EvaluateResultRequest, ListDataSetRequest, LoadDataRequest, LoraRequest, MergeModelsRequest, TrainConfigRequest
from src.train.config_manager import TrainConfigManager
from src.train.data_loader import MultiFormatDataLoader
from src.train.train_engine import TrainEngine, merge_lora_with_base
from src.ext.model_loader import global_model_loader


router = APIRouter(
    prefix="/api/train",
    tags=['模型训练']
)

# 全局状态标志
is_training = False
training_lock = asyncio.Lock()


@router.post("/load_dataset")
async def load_dataset(request: LoadDataRequest):
    loader = MultiFormatDataLoader(
            file_paths=request.data_path
        )
    return JSONResponse(loader.load_data())

@router.post("/list_dataset")   
async def list_dataset(request: ListDataSetRequest):
    
    if not os.path.exists(request.data_dir):
        raise HTTPException("data_dir is not exists")
    
    if not os.path.isdir(request.data_dir):
        raise HTTPException("data_dir is not a directory")
    
    files_info = []
    try:
        all_entries = os.listdir(request.data_dir)
        for entry_name in all_entries:
            full_path = os.path.join(request.data_dir, entry_name)
            files_info.append({"name": entry_name, "data_path": full_path})
        
    except Exception as e:
        raise HTTPException(f"list dataset error: {e}")
    
    return JSONResponse(files_info)
    
@router.post("/load_lora_config")
async def load_lora_config(request:LoraRequest):

    # 你的 LoRA 适配器路径
    lora_path = request.lora_path

    # 配置文件通常名为 adapter_config.json
    config_file_path = os.path.join(lora_path, "adapter_config.json")

    if os.path.exists(config_file_path):
        with open(config_file_path, 'r') as f:
            config_data = json.load(f)

        # 直接从 JSON 字典中获取参数
        lora_rank = config_data['r']
        lora_alpha = config_data['lora_alpha']
        target_modules = config_data['target_modules']
        lora_dropout = config_data['lora_dropout']

        return JSONResponse({
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "target_modules": target_modules,
            "lora_dropout": lora_dropout
        })

    raise RequestValidationError([{"loc": ["lora_path"], "msg": "LoRA 路径不存在"}])

@router.post("/start_train")
async def start_train(request: TrainConfigRequest):

    global is_training
    async with training_lock:
        if is_training:
            raise Exception("Train or evaluate_loss is already running")
        is_training = True
    try:
        #卸载全局的tokenizer和model
        global_model_loader.unload_tokenizer()
        global_model_loader.unload_model()

        # 在这里放置你真正的训练代码
        config_dict = request.model_dump()
        # 为每个训练添加一个时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_dict["output_dir"] = os.path.join(config_dict["output_dir"], f"train_{timestamp}")
        config = TrainConfigManager.register_from_dict(config_dict)
        if config.use_deepspeed:
            # 请使用脚本启动 deepspeed 训练
            # raise Exception("Please use the script to start deepspeed training.")
        
            # 使用 subprocess 启动 deepspeed 训练
            import subprocess
            import sys
            from src.utils.misc import find_available_port,get_device_count
            device_count = get_device_count()
            if device_count > 0:
                device_list = ",".join(str(i) for i in range(device_count))
                include_arg = f"localhost:{device_list}"
            else:
                include_arg = "localhost:0"  # 默认使用第一个设备
    
            deepspeed_cmd = [
                "deepspeed",
                "--master_port", str(find_available_port()),
                "--include",include_arg,
                "--module", "start_training",
                "--model_path_to_load", config.model_path_to_load,
                "--data_type", "bfloat16" if config.data_type==torch.bfloat16 else "float16" if config.data_type==torch.float16 else "float32",
                "--use_lora", str(config.use_lora).lower(),
                "--lora_rank", str(config.lora_rank),
                "--lora_alpha", str(config.lora_alpha),
                "--lora_dropout", str(config.lora_dropout),
                "--lora_target_modules", ','.join(item for item in config.lora_target_modules),
                "--data_path", ','.join(item for item in config.data_path),
                "--training_type", str(config.training_type.value),
                "--enable_gradit_checkpoing", str(config.gradit_checkpoing).lower(),
                "--epoch", str(config.epoch),
                "--lr", str(config.lr),
                "--batch_size", str(config.batch_size),
                "--gradient_accumulation_steps", str(config.gradient_accumulation_steps),
                "--max_seq_len", str(config.max_seq_len),
                '--use_deepspeed', str(config.use_deepspeed).lower(),
                "--pack_length", str(config.pack_length),
                "--use_NLIRG", str(config.use_nlirg).lower(),
                "--token_batch", str(config.token_batch),
                "--output_dir", str(config.output_dir),
                "--use_tensorboard", str(config.use_tensorboard).lower(),
                ]

            if len(config.checkpoint_epoch)>0:
                deepspeed_cmd.append("--checkpoint_epoch")
                deepspeed_cmd.append(','.join(item for item in config.checkpoint_epoch))
            
            deepspeed_cmd.append("--system_prompt")
            deepspeed_cmd.append(f'"{config.system_prompt}"')

            if config.lora_path:
                deepspeed_cmd.append("--lora_path")
                deepspeed_cmd.append(config.lora_path)
            if config.tensorboard_path:
                deepspeed_cmd.append("--tensorboard_path")
                deepspeed_cmd.append(config.tensorboard_path)


            # 启动 deepspeed 进程
            process = subprocess.Popen(
                " ".join(deepspeed_cmd),  # 将列表转换为字符串
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )


            # 实时输出日志
            for line in process.stdout:
                print(line, end='')
            is_training = False

        else:

            # 启动训练的后台任务
            async def run_training():
                """
                实际的训练任务函数。
                """
                try:
                    await asyncio.to_thread(lambda: TrainEngine(config).start_train())
                finally:
                    global is_training
                    is_training = False

            asyncio.create_task(run_training())

        return JSONResponse({"message": "Training started"})
    
    except Exception as e:
        is_training = False
        raise Exception(str(e))

@router.post("/evaluate_loss")
async def eval_loss(request: EvaluateLossRequest = Body(...)):
    """
    评估模型损失
    
    接收一个 JSON 对象作为请求体，包含评估所需的所有参数。
    """
    global is_training
    async with training_lock:
        if is_training:
            raise Exception("Train or evaluate_loss is already running")
        is_training = True

    try:
        #卸载全局的tokenizer和model
        global_model_loader.unload_tokenizer()
        global_model_loader.unload_model()
        import uuid

        dir_name, file_name = os.path.split(request.data_path)
        stem, ext = os.path.splitext(file_name)
        new_file_name = f"{stem}_{uuid.uuid4().hex}{ext}"
        output_path = os.path.join(dir_name, new_file_name)
        # 确保输出文件总是有.json扩展名
        if not output_path.endswith('.json'):
            if output_path.endswith('.'):
                output_path = output_path[:-1] + '.json'
            else:
                output_path += '.json'
        async def run_evaluate_loss():
            """
            实际的训练任务函数。
            """
            try:
                await asyncio.to_thread(lambda: evaluate_loss(
                        data_path=request.data_path,
                        model_path=request.model_path,
                        output_path=output_path,
                        method=request.method,
                        sort=False
                    ))
            finally:
                global is_training
                is_training = False

        asyncio.create_task(run_evaluate_loss())

        return JSONResponse({"output_path": output_path})
    
    except Exception as e:
        is_training = False
        raise Exception(str(e))

@router.post("/evaluate_result")   
async def evaluate_result(request: EvaluateResultRequest):
    """
    评估模型损失
    
    接收一个 JSON 对象作为请求体，包含评估所需的所有参数。
    """
    if not os.path.exists(request.output_path):
        raise Exception("评估任务完成后，才能获取评估结果")
    try:
        with open(request.output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return JSONResponse(data)
    except Exception as e:
        raise Exception(str(e))

@router.post("/merge_models")
async def merge_models(request: MergeModelsRequest = Body(...)):
    """
    合并模型
    
    接收一个 JSON 对象作为请求体，包含合并所需的所有参数。
    """
    #卸载全局的tokenizer和model
    global_model_loader.unload_tokenizer()
    global_model_loader.unload_model()

    merge_lora_with_base(
        base_model_path=request.base_model_path,
        lora_path=request.lora_path,
        output_path=request.output_path
    )
    return {"message": "Models merged successfully"}

@router.websocket("/ws/terminal")
async def websocket_endpoint(websocket: WebSocket):

    from src.api.custom_output import terminal_collector

    await terminal_collector.add_ws(websocket)
    try:
        while True:
            # 保持连接
            await websocket.receive_text()
    except WebSocketDisconnect:
        terminal_collector.remove_ws(websocket)
    except Exception as e:
        print(f"WebSocket错误: {e}")
        terminal_collector.remove_ws(websocket)


