from fastapi import APIRouter, Body, UploadFile, Form
from fastapi.exceptions import RequestValidationError

from src.configs.server_config import  MODEL_DIR
from src.ext.model_loader import global_model_loader
from src.utils.file_helper import FileHelper

router = APIRouter(
    prefix="/api/base",
    tags=['基础服务']
)


@router.get("/current_model_name", summary="当前模型名称")
def current_model_name():
    model = global_model_loader.load_model()
    model_name = model.config.name_or_path
    return {"model_name": model_name}


@router.get("/get_models", summary="获取所有模型")
def get_models():
    return {"model_name": FileHelper.get_file_paths(MODEL_DIR)}


@router.get("/change_model", summary="切换模型")
def change_model(model_name: str):
    global_model_loader.switch_model(model_name)
    return {"message": "模型已切换"}
