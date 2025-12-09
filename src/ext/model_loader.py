import gc
import os
import threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.configs.server_config import MODEL_DIR
from src.utils.file_helper import FileHelper

class ModelLoader:
    _instance = None  # 类变量，用于存储单例实例
    _model = None     # 类变量，用于存储加载的模型
    _tokenizer = None # 类变量，用于存储加载的tokenizer
    _lock = threading.Lock()  # 类级锁，用于控制模型切换

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path,device_map="auto",torch_dtype=torch.bfloat16,attn_implementation="eager"):
        if not hasattr(self, '_initialized'):  # 避免重复初始化
            self.model_path = model_path
            self.torch_dtype=torch_dtype
            self.device_map=device_map
            self.attn_implementation=attn_implementation
            self._initialized = True

    def load_model(self):
        if self._model is None:
            if not os.path.exists(self.model_path):
                raise Exception(f"模型路径不存在: {self.model_path}")
            if not any(os.listdir(self.model_path)):
                raise Exception(f"模型路径为空，请先保存模型到->{self.model_path}")

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                device_map=self.device_map,
                torch_dtype=self.torch_dtype,
                attn_implementation=self.attn_implementation
            )
        return self._model
    
    def load_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                device_map=self.device_map
            )
        return self._tokenizer

    def unload_model(self):
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()

    def unload_tokenizer(self):
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
            gc.collect()

    def switch_model(self, model_path):
        with self._lock:  # 加锁，防止并发切换
            self.unload_model()
            self.unload_tokenizer()
            self.model_path = model_path
            self.load_model()
            self.load_tokenizer()

    def __enter__(self):
        self.load_model()
        self.load_tokenizer()
        return self._model, self._tokenizer

    def __exit__(self, exctype, excval, exctb):
        self.unload_model()
        self.unload_tokenizer()

def get_global_model_loader():
    try:
        return ModelLoader(FileHelper.get_file_paths(MODEL_DIR)[0])
    except Exception as e:
        print(f"获取全局模型加载器失败: {e}")
        return None
    

global_model_loader = get_global_model_loader()



