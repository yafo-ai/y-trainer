import json
import csv
import os
import glob
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum


class DatasetFormat(Enum):
    """支持的数据集格式"""
    ALPACA = "alpaca"          # Alpaca格式: instruction, input, output
    CONVERSATION = "conversation"  # 对话格式: conversations字段
    SHAREGPT = "sharegpt"      # ShareGPT格式: messages字段
    BELLE = "belle"           # BELLE格式: instruction, output
    DIALOGUE = "dialogue"      # 多轮对话格式
    PROMPT_RESPONSE = "prompt_response"  # 简单的prompt-response对
    CUSTOM = "custom"          # 自定义格式


@dataclass
class DataItem:
    """统一的数据项格式"""
    system: str = ""
    instruction: str = ""
    input: str = ""
    output: str = ""
    conversations: List[Dict] = None  # 对话格式: [{"role": "user", "content": "..."}, ...]
    history: List[Dict] = None        # 历史对话: [[q1, a1], [q2, a2], ...]
    id: Any = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.conversations is None:
            self.conversations = []
        if self.history is None:
            self.history = []
        if self.metadata is None:
            self.metadata = {}


class MultiFormatDataLoader:
    """支持多种格式的数据加载器"""
    
    def __init__(self, 
                 file_paths: Union[str, List[str]],
                 system_prompt: str = "",
                 format_type: Optional[DatasetFormat] = None,
                 max_samples: Optional[int] = None,
                 skip_empty: bool = False,
                 encoding: str = 'utf-8'):
        """
        初始化数据加载器
        
        Args:
            file_paths: 文件路径，可以是单个文件、文件列表或目录
            system_prompt: 系统提示词
            format_type: 数据集格式，如果为None则自动检测
            max_samples: 最大加载样本数
            skip_empty: 是否跳过空样本
            encoding: 文件编码
        """
        self.system_prompt = system_prompt
        self.format_type = format_type
        self.max_samples = max_samples
        self.skip_empty = skip_empty
        self.encoding = encoding
        
        # 处理文件路径
        if isinstance(file_paths, str):
            if os.path.isdir(file_paths):
                # 如果是目录，查找所有支持的文件
                self.file_paths = self._find_data_files(file_paths)
            else:
                self.file_paths = [file_paths]
        else:
            self.file_paths = file_paths
    
    def _find_data_files(self, directory: str) -> List[str]:
        """在目录中查找数据文件"""
        supported_extensions = ['*.json', '*.jsonl', '*.csv', '*.txt']
        files = []
        for ext in supported_extensions:
            files.extend(glob.glob(os.path.join(directory, ext)))
        return sorted(files)
    
    def _detect_format(self, data: Any) -> DatasetFormat:
        """自动检测数据格式"""
        if isinstance(data, list):
            if len(data) == 0:
                return DatasetFormat.CUSTOM
            
            # 检查第一个样本的格式
            sample = data[0]
            
            if 'conversations' in sample or 'messages' in sample:
                return DatasetFormat.CONVERSATION
            elif 'instruction' in sample and 'output' in sample:
                if 'input' in sample:
                    return DatasetFormat.ALPACA
                else:
                    return DatasetFormat.BELLE
            elif 'prompt' in sample and 'response' in sample:
                return DatasetFormat.PROMPT_RESPONSE
            elif 'history' in sample or ('question' in sample and 'answer' in sample):
                return DatasetFormat.DIALOGUE
             
        
        return DatasetFormat.CUSTOM
    
    def _load_json(self, file_path: str) -> List[Dict]:
        """加载JSON文件"""
        with open(file_path, 'r', encoding=self.encoding) as f:
            return json.load(f)
    
    def _load_jsonl(self, file_path: str) -> List[Dict]:
        """加载JSONL文件"""
        data = []
        with open(file_path, 'r', encoding=self.encoding) as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def _load_csv(self, file_path: str) -> List[Dict]:
        """加载CSV文件"""
        data = []
        with open(file_path, 'r', encoding=self.encoding) as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(dict(row))
        return data
    
    def _load_txt(self, file_path: str) -> List[Dict]:
        """加载文本文件（每行一个样本）"""
        data = []
        with open(file_path, 'r', encoding=self.encoding) as f:
            for idx, line in enumerate(f):
                if line.strip():
                    data.append({
                        'output': line.strip(),
                        'id': idx
                    })
        return data
    
    def _parse_alpaca_format(self, item: Dict, idx: int) -> Optional[DataItem]:
        """解析Alpaca格式"""
        id = item.get("id", idx)
        if isinstance(id, str):
            try:
                id = int(id)
            except (ValueError, TypeError):
                id = idx
        
        data_item = DataItem(
            system=self.system_prompt,
            instruction=item.get("instruction", ""),
            input=item.get("input", ""),
            output=item.get("output", ""),
            id=id,
            metadata={k: v for k, v in item.items() 
                     if k not in ['instruction', 'input', 'output', 'id']}
        )
        
        if self.skip_empty:
            if not data_item.output or (not data_item.instruction and not data_item.input):
                return None
        
        return data_item
    
    def _parse_conversation_format(self, item: Dict, idx: int) -> Optional[DataItem]:
        """解析对话格式"""
        conversations = item.get('conversations', item.get('messages', []))
        
        # 将对话格式转换为instruction/output格式
        if conversations:
            # 提取用户和助手的对话
            user_messages = []
            assistant_messages = []
            
            for conv in conversations:
                role = conv.get('role', conv.get('from', '')).lower()
                content = conv.get('content', conv.get('value', ''))
                
                if role in ['user', 'human']:
                    user_messages.append(content)
                elif role in ['assistant', 'gpt', 'bot']:
                    assistant_messages.append(content)
            
            # 如果对话是多轮的，可以处理为多轮历史
            if len(user_messages) > 1 and len(assistant_messages) > 1:
                history = []
                for i in range(min(len(user_messages), len(assistant_messages))):
                    history.append([user_messages[i], assistant_messages[i]])
                
                data_item = DataItem(
                    system=self.system_prompt,
                    instruction=user_messages[-1] if user_messages else "",
                    output=assistant_messages[-1] if assistant_messages else "",
                    conversations=conversations,
                    history=history[:-1] if len(history) > 1 else [],
                    id=item.get("id", idx),
                    metadata=item
                )
            else:
                # 单轮对话
                data_item = DataItem(
                    system=self.system_prompt,
                    instruction=user_messages[-1] if user_messages else "",
                    output=assistant_messages[-1] if assistant_messages else "",
                    conversations=conversations,
                    id=item.get("id", idx),
                    metadata=item
                )
            
            if self.skip_empty and not data_item.output:
                return None
            
            return data_item
        
        return None
    
    def _parse_sharegpt_format(self, item: Dict, idx: int) -> Optional[DataItem]:
        """解析ShareGPT格式"""
        return self._parse_conversation_format(item, idx)
    
    def _parse_belle_format(self, item: Dict, idx: int) -> Optional[DataItem]:
        """解析BELLE格式"""
        data_item = DataItem(
            system=self.system_prompt,
            instruction=item.get("instruction", ""),
            output=item.get("output", ""),
            id=item.get("id", idx),
            metadata=item
        )
        
        if self.skip_empty and (not data_item.output or not data_item.instruction):
            return None
        
        return data_item
    
    def _parse_prompt_response_format(self, item: Dict, idx: int) -> Optional[DataItem]:
        """解析prompt-response格式"""
        data_item = DataItem(
            system=self.system_prompt,
            instruction=item.get("prompt", ""),
            output=item.get("response", item.get("completion", "")),
            id=item.get("id", idx),
            metadata=item
        )
        
        if self.skip_empty and (not data_item.output or not data_item.instruction):
            return None
        
        return data_item
    
    def _parse_dialogue_format(self, item: Dict, idx: int) -> Optional[DataItem]:
        """解析多轮对话格式"""
        history = item.get('history', [])
        if not history and 'question' in item and 'answer' in item:
            # 如果是Q&A格式，可以转换为单轮
            data_item = DataItem(
                system=self.system_prompt,
                instruction=item.get("question", ""),
                output=item.get("answer", ""),
                id=item.get("id", idx),
                metadata=item
            )
        else:
            # 多轮历史对话
            data_item = DataItem(
                system=self.system_prompt,
                instruction=item.get("instruction", item.get("query", "")),
                output=item.get("response", item.get("output", "")),
                history=history,
                id=item.get("id", idx),
                metadata=item
            )
        
        if self.skip_empty and not data_item.output:
            return None
        
        return data_item
    
    def _parse_custom_format(self, item: Dict, idx: int) -> Optional[DataItem]:
        """解析自定义格式"""
        # 尝试自动推断字段
        instruction = item.get("instruction", item.get("query", item.get("question", "")))
        output = item.get("output", item.get("response", item.get("answer", item.get("completion", item.get("text", "")))))
        
        data_item = DataItem(
            system=self.system_prompt,
            instruction=instruction,
            input=item.get("input", ""),
            output=output,
            id=item.get("id", idx),
            metadata=item
        )
        
        if self.skip_empty and (not data_item.output or (not data_item.instruction and not data_item.input)):
            return None
        
        return data_item
    
    def load_data(self) -> List[Dict]:
        """加载所有数据"""
        all_data = []
        
        for file_idx, file_path in enumerate(self.file_paths):
            try:
                # 根据文件扩展名选择加载方法
                ext = os.path.splitext(file_path)[1].lower()
                
                if ext == '.json':
                    raw_data = self._load_json(file_path)
                elif ext == '.jsonl':
                    raw_data = self._load_jsonl(file_path)
                elif ext == '.csv':
                    raw_data = self._load_csv(file_path)
                elif ext == '.txt' or ext == '.tsv':
                    raw_data = self._load_txt(file_path)
                else:
                    print(f"Unsupported file format: {ext}")
                    continue
                
                # 自动检测格式（如果未指定）
                format_type = self.format_type
                if format_type is None:
                    format_type = self._detect_format(raw_data)
                
                # 解析数据
                parser_map = {
                    DatasetFormat.ALPACA: self._parse_alpaca_format,
                    DatasetFormat.CONVERSATION: self._parse_conversation_format,
                    DatasetFormat.SHAREGPT: self._parse_sharegpt_format,
                    DatasetFormat.BELLE: self._parse_belle_format,
                    DatasetFormat.PROMPT_RESPONSE: self._parse_prompt_response_format,
                    DatasetFormat.DIALOGUE: self._parse_dialogue_format,
                    DatasetFormat.CUSTOM: self._parse_custom_format,
                }
                
                parser = parser_map.get(format_type, self._parse_custom_format)
                
                for idx, item in enumerate(raw_data):
                    if self.max_samples and len(all_data) >= self.max_samples:
                        break
                    
                    try:
                        data_item = parser(item, idx)
                        if data_item is not None:
                            # 如果没有data_id，使用全局索引
                            if data_item.id is None:
                                data_item.id = len(all_data)
                            all_data.append(data_item)
                    except Exception as e:
                        print(f"Error parsing item {idx} in file {file_path}: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"Failed to load data from file {file_path}: {str(e)}")
                continue
        
        return self.to_dict_list(all_data)
    
    def to_dict_list(self, data_items: List[DataItem]) -> List[Dict]:
        """将DataItem列表转换为字典列表"""
        def is_not_empty(value):
            """检查值是否非空"""
            if value is None:
                return False
            if isinstance(value, str) and value.strip() == "":
                return False
            if isinstance(value, (list, dict, set)) and len(value) == 0:
                return False
            return True
        
        result = []
        for item in data_items:
            item_dict = {}
            
            # 添加字段（如果非空）
            item_dict["system"] = item.system
            item_dict["instruction"] = item.instruction
            item_dict["input"] = item.input
            item_dict["output"] = item.output
            item_dict["data_id"] = item.id
            result.append(item_dict)
        
        return result


# # 使用示例
# def example_usage():
#     # 示例1：加载Alpaca格式数据
#     loader = MultiFormatDataLoader(
#         file_paths="data/sharegpt_data.jsonl",
#         system_prompt="You are a helpful assistant.",
#     )
#     data = loader.load_data()
    
#     # 示例2：自动检测格式
#     loader = MultiFormatDataLoader(
#         file_paths=["data/cpt_data.jsonl"],
#         system_prompt="You are a helpful assistant."
#     )
#     data = loader.load_data()
    
#     # 示例3：加载目录中的所有数据文件
#     loader = MultiFormatDataLoader(
#         file_paths="data/conversation_data.json",
#         system_prompt="You are a helpful assistant.",
#         max_samples=1000  # 只加载1000个样本
#     )
#     data = loader.load_data()

#     print(f"Loaded {len(data)} samples.")
    
# if __name__ == "__main__":
#     example_usage()