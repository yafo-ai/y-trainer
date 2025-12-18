from fastapi import APIRouter, Body, UploadFile, Form
from fastapi.exceptions import RequestValidationError

from src.ext.model_loader import global_model_loader
from src.utils.file_helper import FileHelper

router = APIRouter(
    prefix="/api/base",
    tags=['基础服务']
)


@router.get("/current_model_name", summary="当前模型名称")
def current_model_name():
   
   return global_model_loader.get_current_mdoel()


@router.get("/change_model", summary="加载切换模型")
def change_model(model_name: str,lora_path:str=""):
    global_model_loader.switch_model(model_name,lora_path)
    return {"message": "模型加载完成"}


#卸载模型
@router.get("/unload_model", summary="卸载模型")
def unload_model():
    global_model_loader.unload_model()
    global_model_loader.unload_tokenizer()
    return {"message": "模型卸载完成"}
