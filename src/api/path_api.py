from fastapi import APIRouter, Query
from pathlib import Path
import os
from typing import List, Optional
from pydantic import BaseModel
from enum import Enum


router = APIRouter(
    prefix="/api/path",
    tags=['路径浏览API']
)



class SearchResult(BaseModel):
    name: str
    type: str  # 'file' or 'directory'
    full_path: str



@router.get("/search", response_model=List[SearchResult], summary="搜索路径建议")
async def search_path(
    query: str = Query("", description="搜索查询字符串"),
    target_type: str = Query("all", description="搜索目标类型: directory-文件夹, file-文件, all-所有", regex="^(directory|file|all)$")
):
    """
    根据查询字符串搜索路径建议
    
    搜索规则：
    1. 如果查询字符串指向一个存在的文件夹，则在该文件夹内搜索其子文件和子文件夹
    2. 如果查询字符串不存在或指向文件，则在父目录中搜索匹配项
    3. 如果查询字符串是部分路径，则在可能存在的父目录中搜索
    
    根据target_type参数过滤结果：
    - directory: 只搜索文件夹，如果没有子目录则加载同级目录
    - file: 只搜索文件，如果没有子文件夹则展示同目录下的文件
    - all: 搜索所有类型的文件
    
    新增功能：
    - 当target_type="file"且精确匹配到一个文件时，返回该文件的同级兄弟文件
    - 当target_type="file"且输入路径是完整文件路径时，返回同级文件
    
    示例：
    - /home/user/documents → 在 /home/user/documents 文件夹内搜索所有子项
    - /home/user/doc → 在 /home/user 中搜索以 "doc" 开头的项
    - projects/src → 如果 projects/src 存在且是文件夹，则在该文件夹内搜索
    - myfolder/ → 在 myfolder 文件夹内搜索（如果存在）
    """
    
    # 如果查询字符串为空，返回空列表
    if not query.strip():
        return []
    
    # 规范化路径分隔符
    normalized_query = query.replace("\\", "/")
    
    # 尝试判断查询字符串是否指向一个存在的文件夹或文件
    is_existing_folder = False
    is_existing_file = False
    target_folder = ""
    target_file = ""
    
    try:
        # 检查查询路径是否存在
        test_path = Path(normalized_query)
        
        # 处理以斜杠结尾的情况
        if normalized_query.endswith("/"):
            # 去掉尾部的斜杠后检查
            test_path_no_slash = Path(normalized_query.rstrip("/"))
            if test_path_no_slash.exists() and test_path_no_slash.is_dir():
                is_existing_folder = True
                target_folder = str(test_path_no_slash)
            else:
                # 可能是一个不存在的路径，按普通搜索处理
                pass
        else:
            # 普通路径，检查是否存在
            if test_path.exists():
                if test_path.is_dir():
                    is_existing_folder = True
                    target_folder = normalized_query
                elif test_path.is_file():
                    is_existing_file = True
                    target_file = normalized_query
    except Exception:
        # 路径无效，按普通搜索处理
        pass
    
    # 根据不同的情况处理搜索
    if is_existing_folder:
        # 情况1：查询字符串指向一个存在的文件夹，搜索其内容
        return await search_inside_folder(target_folder, query, target_type)
    elif is_existing_file and target_type == "file":
        # 情况2：查询字符串指向一个存在的文件，且目标类型为文件
        # 返回该文件的同级兄弟文件
        return await search_sibling_files(target_file, query, target_type)
    else:
        # 情况3：普通搜索，在父目录中搜索匹配项
        return await search_in_parent_directory(normalized_query, query, target_type)


async def search_sibling_files(file_path: str, original_query: str, target_type: str = "file") -> List[SearchResult]:
    """
    搜索指定文件的同级兄弟文件
    
    参数:
        file_path: 存在的文件路径
        original_query: 原始查询字符串
        target_type: 搜索目标类型（这里应为file）
    
    返回:
        同级目录下的所有文件（包括当前文件）
    """
    try:
        file_path_obj = Path(file_path)
        
        # 确保路径存在且是文件
        if not file_path_obj.exists() or not file_path_obj.is_file():
            return []
        
        # 获取文件所在目录
        parent_dir = file_path_obj.parent
        
        # 判断原始查询是否为绝对路径
        is_absolute = os.path.isabs(original_query) or original_query.startswith(('/', '\\'))
        
        # 收集结果
        results = []
        
        # 获取目录内的所有文件
        try:
            items = list(parent_dir.iterdir())
            
            # 先添加当前文件（作为第一个结果）
            current_file_type = "directory" if file_path_obj.is_dir() else "file"
            
            # 构建当前文件的返回路径
            if is_absolute:
                current_file_path = str(file_path_obj.absolute())
            else:
                # 保持原始查询格式
                current_file_path = original_query
            
            results.append({
                "name": file_path_obj.name,
                "type": current_file_type,
                "full_path": current_file_path,
                "_score": 100,  # 当前文件评分最高
                "_is_current": True  # 标记为当前文件
            })
            
            # 添加同级文件
            for item in items:
                try:
                    # 跳过隐藏文件和当前文件
                    if item.name.startswith('.') or item.name == file_path_obj.name:
                        continue
                    
                    # 根据target_type过滤
                    item_type = "directory" if item.is_dir() else "file"
                    
                    if target_type == "file" and item_type != "file":
                        # 只显示文件
                        continue
                    elif target_type == "directory" and item_type != "directory":
                        # 只显示文件夹（虽然不太可能在这里出现）
                        continue
                    
                    # 构建返回路径
                    if is_absolute:
                        return_path = str(item.absolute())
                    else:
                        # 构建相对于父目录的路径
                        # 从原始查询中提取父目录部分
                        parts = original_query.rstrip("/").split("/")
                        if len(parts) > 1:
                            parent_query = "/".join(parts[:-1])
                            if parent_query:
                                return_path = f"{parent_query}/{item.name}"
                            else:
                                return_path = item.name
                        else:
                            return_path = item.name
                    
                    results.append({
                        "name": item.name,
                        "type": item_type,
                        "full_path": return_path,
                        "_score": 80,  # 同级文件评分
                        "_is_current": False
                    })
                    
                except (PermissionError, OSError):
                    continue
                    
        except (PermissionError, OSError):
            # 无权限访问目录内容
            return []
        
        # 按名称排序（但当前文件保持第一）
        current_file = results[0]
        other_files = sorted(results[1:], key=lambda x: x["name"].lower())
        
        # 合并结果
        sorted_results = [current_file] + other_files
        
        # 转换为SearchResult对象
        search_results = [
            SearchResult(
                name=item["name"],
                type=item["type"],
                full_path=item["full_path"]
            ) for item in sorted_results
        ]
        
        return search_results
        
    except Exception as e:
        # 发生错误，返回空列表
        return []


async def search_in_parent_directory(normalized_query: str, original_query: str, target_type: str = "all") -> List[SearchResult]:
    """
    在父目录中搜索匹配项（普通搜索模式）
    
    参数:
        normalized_query: 规范化后的查询字符串
        original_query: 原始查询字符串
        target_type: 搜索目标类型: directory/file/all
    
    返回:
        匹配的搜索结果
    """
    # 判断是否为绝对路径
    is_absolute = os.path.isabs(original_query) or original_query.startswith(('/', '\\'))
    
    # 提取搜索关键词和父目录
    if "/" in normalized_query:
        # 包含路径分隔符，提取最后一部分和父目录
        parts = normalized_query.rstrip("/").split("/")
        if len(parts) > 1:
            search_keyword = parts[-1]
            parent_path = "/".join(parts[:-1])
            
            # 如果父路径为空且是绝对路径，使用根目录
            if not parent_path and is_absolute:
                parent_path = "/"
        else:
            # 只有一个部分，但包含斜杠（例如 "/doc"）
            search_keyword = parts[0] if parts else ""
            parent_path = "/" if is_absolute else "."
    else:
        # 纯名称，在默认目录中搜索
        search_keyword = normalized_query
        parent_path = "."  # 相对路径使用当前目录
    
    # 处理父目录路径
    try:
        if is_absolute:
            parent_dir = Path(parent_path).resolve()
        else:
            parent_dir = Path(parent_path)
            if not parent_dir.is_absolute():
                parent_dir = Path.cwd() / parent_dir
        
        # 确保父目录存在
        if not parent_dir.exists() or not parent_dir.is_dir():
            # 尝试查找存在的父目录
            current_path = Path(parent_path)
            while str(current_path) != "." and str(current_path) != "/" and not current_path.exists():
                current_path = current_path.parent
            
            if current_path.exists() and current_path.is_dir():
                parent_dir = current_path
            else:
                # 如果找不到存在的父目录，使用当前目录
                parent_dir = Path.cwd()
                
    except Exception:
        # 路径无效，使用当前目录
        parent_dir = Path.cwd()
    
    # 检查是否精确匹配到一个文件（针对target_type="file"的情况）
    if target_type == "file":
        # 检查父目录中是否存在与搜索关键词完全匹配的文件
        exact_file_match = None
        try:
            # 构建可能的文件路径
            possible_file_path = parent_dir / search_keyword
            if possible_file_path.exists() and possible_file_path.is_file():
                exact_file_match = possible_file_path
        except Exception:
            pass
        
        if exact_file_match:
            # 精确匹配到文件，返回该文件的同级文件
            # 构建原始查询的完整路径
            if is_absolute:
                exact_file_query = str(exact_file_match.absolute())
            else:
                exact_file_query = f"{parent_path}/{search_keyword}" if parent_path != "." else search_keyword
            
            return await search_sibling_files(str(exact_file_match), exact_file_query, target_type)
    
    # 在父目录中搜索
    results = []
    
    # 根据target_type决定搜索策略
    if target_type == "directory":
        # 只搜索文件夹
        results = await search_directories_in_parent(parent_dir, search_keyword, parent_path, original_query, is_absolute)
        
        # 如果没有找到文件夹且父目录不是根目录，尝试搜索同级目录
        if not results and parent_dir != Path("/") and parent_dir != Path(".").resolve():
            parent_parent = parent_dir.parent
            # 搜索父目录的同级目录
            try:
                for item in parent_parent.iterdir():
                    try:
                        if item.name.startswith('.') or item.name == parent_dir.name:
                            continue
                        
                        if item.is_dir():
                            # 构建返回路径
                            if is_absolute:
                                return_path = str(item.absolute())
                            else:
                                # 计算相对路径
                                rel_path = os.path.relpath(item, Path.cwd())
                                return_path = str(rel_path) if not rel_path.startswith("..") else str(item)
                            
                            results.append({
                                "name": item.name,
                                "type": "directory",
                                "full_path": return_path,
                                "_score": 30  # 同级目录，评分较低
                            })
                    except (PermissionError, OSError):
                        continue
            except (PermissionError, OSError):
                pass
    
    elif target_type == "file":
        # 先搜索匹配的文件夹
        folder_results = []
        file_results = []
        
        try:
            for item in parent_dir.iterdir():
                try:
                    # 检查是否匹配搜索关键词
                    if search_keyword and search_keyword.lower() in item.name.lower():
                        match_score = calculate_match_score(item.name, search_keyword)
                        
                        # 构建返回路径
                        return_path = build_return_path(item, parent_path, original_query, is_absolute)
                        
                        if item.is_dir():
                            folder_results.append({
                                "name": item.name,
                                "type": "directory",
                                "full_path": return_path,
                                "_score": match_score
                            })
                        else:
                            file_results.append({
                                "name": item.name,
                                "type": "file",
                                "full_path": return_path,
                                "_score": match_score
                            })
                except (PermissionError, OSError):
                    continue
        except (PermissionError, OSError):
            return []
        
        # 优先返回文件夹（方便导航）
        if folder_results:
            results = folder_results
        else:
            # 如果没有文件夹，返回匹配的文件
            results = file_results
            
            # 如果匹配的文件只有一个，返回该文件的同级文件（便于筛选）
            if len(results) == 1 and results[0]["type"] == "file":
                # 构建文件的完整路径
                matched_file = results[0]
                try:
                    # 从full_path构建Path对象
                    if is_absolute:
                        file_path = Path(matched_file["full_path"])
                    else:
                        # 尝试解析相对路径
                        file_path = (Path.cwd() / parent_path / Path(matched_file["full_path"]).name).resolve()
                    
                    if file_path.exists() and file_path.is_file():
                        return await search_sibling_files(str(file_path), matched_file["full_path"], target_type)
                except Exception:
                    pass
    
    else:  # target_type == "all"
        # 搜索所有类型
        try:
            for item in parent_dir.iterdir():
                try:
                    # 检查是否匹配搜索关键词
                    if search_keyword and search_keyword.lower() in item.name.lower():
                        match_score = calculate_match_score(item.name, search_keyword)
                        
                        # 构建返回路径
                        return_path = build_return_path(item, parent_path, original_query, is_absolute)
                        
                        results.append({
                            "name": item.name,
                            "type": "directory" if item.is_dir() else "file",
                            "full_path": return_path,
                            "_score": match_score
                        })
                except (PermissionError, OSError):
                    continue
        except (PermissionError, OSError):
            return []
    
    # 按匹配度排序
    results.sort(key=lambda x: x["_score"], reverse=True)
    
    # 限制结果数量
    max_results = 50
    
    return [
        SearchResult(
            name=item["name"],
            type=item["type"],
            full_path=item["full_path"]
        ) for item in results[:max_results]
    ]


async def search_inside_folder(folder_path: str, original_query: str, target_type: str = "all") -> List[SearchResult]:
    """
    在指定的文件夹内搜索其子文件和子文件夹
    
    参数:
        folder_path: 要搜索的文件夹路径
        original_query: 原始查询字符串（用于保持路径格式）
        target_type: 搜索目标类型: directory/file/all
    
    返回:
        文件夹内的所有子项
    """
    try:
        path = Path(folder_path)
        
        # 确保路径存在且是文件夹
        if not path.exists() or not path.is_dir():
            return []
        
        # 判断原始查询是否为绝对路径
        is_absolute = os.path.isabs(original_query) or original_query.startswith(('/', '\\'))
        
        # 收集结果
        results = []
        
        # 获取文件夹内的所有项目
        try:
            items = list(path.iterdir())
            
            # 限制最大结果数量
            max_results = 50
            
            # 先收集所有符合条件的项目
            temp_results = []
            for item in items[:max_results]:
                try:
                    # 跳过隐藏文件（以点开头）
                    if item.name.startswith('.'):
                        continue
                    
                    # 根据target_type过滤
                    item_type = "directory" if item.is_dir() else "file"
                    
                    if target_type == "directory" and item_type != "directory":
                        continue
                    elif target_type == "file" and item_type != "file":
                        continue
                    
                    # 构建返回路径
                    if is_absolute:
                        return_path = str(item.absolute())
                    else:
                        # 构建相对路径
                        if original_query.endswith("/"):
                            # 如果原始查询以斜杠结尾，直接附加文件名
                            return_path = f"{original_query}{item.name}"
                        else:
                            # 添加斜杠分隔符
                            return_path = f"{original_query}/{item.name}"
                    
                    temp_results.append({
                        "name": item.name,
                        "type": item_type,
                        "full_path": return_path,
                        "_score": 100  # 文件夹内搜索，所有项评分相同
                    })
                    
                except (PermissionError, OSError):
                    # 跳过无权限访问的项
                    continue
            
            # 检查是否需要扩展搜索范围
            if not temp_results:
                if target_type == "directory":
                    # 搜索文件夹但没有子目录，加载同级目录
                    return await search_sibling_directories(path, original_query, is_absolute)
                elif target_type == "file":
                    # 搜索文件但没有子文件夹，展示当前文件夹内的文件
                    # 已经搜索过所有文件，所以直接返回空列表
                    return []
            
            results = temp_results
                    
        except (PermissionError, OSError):
            # 无权限访问文件夹内容
            if target_type == "directory":
                # 尝试搜索同级目录
                return await search_sibling_directories(path, original_query, is_absolute)
            return []
        
        # 按名称排序
        results.sort(key=lambda x: x["name"].lower())
        
        # 转换为SearchResult对象
        search_results = [
            SearchResult(
                name=item["name"],
                type=item["type"],
                full_path=item["full_path"]
            ) for item in results
        ]
        
        return search_results
        
    except Exception as e:
        # 发生错误，返回空列表
        return []


async def search_sibling_directories(path: Path, original_query: str, is_absolute: bool) -> List[SearchResult]:
    """
    搜索同级目录（父目录下的其他文件夹）
    
    参数:
        path: 当前文件夹路径
        original_query: 原始查询字符串
        is_absolute: 是否为绝对路径
    
    返回:
        同级目录列表
    """
    try:
        parent_path = path.parent
        
        # 如果已经在根目录，无法获取同级目录
        if parent_path == path:
            return []
        
        # 收集父目录下的所有文件夹
        results = []
        try:
            for item in parent_path.iterdir():
                try:
                    # 跳过隐藏文件和当前文件夹
                    if item.name.startswith('.'):
                        continue
                    
                    # 只返回文件夹
                    if item.is_dir():
                        # 构建返回路径
                        if is_absolute:
                            return_path = str(item.absolute())
                        else:
                            # 构建相对于父目录的路径
                            # 需要从original_query中提取父路径
                            parts = original_query.rstrip("/").split("/")
                            if len(parts) > 1:
                                parent_query = "/".join(parts[:-1])
                                if parent_query:
                                    return_path = f"{parent_query}/{item.name}"
                                else:
                                    return_path = item.name
                            else:
                                return_path = item.name
                        
                        results.append({
                            "name": item.name,
                            "type": "directory",
                            "full_path": return_path,
                            "_score": 50  # 同级目录，评分稍低
                        })
                except (PermissionError, OSError):
                    continue
        except (PermissionError, OSError):
            return []
        
        # 按名称排序
        results.sort(key=lambda x: x["name"].lower())
        
        # 转换为SearchResult对象
        search_results = [
            SearchResult(
                name=item["name"],
                type=item["type"],
                full_path=item["full_path"]
            ) for item in results
        ]
        
        return search_results
    except Exception:
        return []


def build_return_path(item: Path, parent_path: str, original_query: str, is_absolute: bool) -> str:
    """
    构建返回路径
    
    参数:
        item: 文件/文件夹项
        parent_path: 父路径
        original_query: 原始查询字符串
        is_absolute: 是否为绝对路径
    
    返回:
        返回路径
    """
    if is_absolute:
        return_path = str(item.absolute())
    else:
        # 构建相对于原始查询父目录的路径
        if parent_path == ".":
            return_path = item.name
        else:
            return_path = f"{parent_path}/{item.name}"
    
    # 确保路径分隔符与原始查询一致
    if "\\" in original_query and "/" in return_path:
        return_path = return_path.replace("/", "\\")
    elif "/" in original_query and "\\" in return_path:
        return_path = return_path.replace("\\", "/")
    
    return return_path


async def search_directories_in_parent(parent_dir: Path, search_keyword: str, parent_path: str, original_query: str, is_absolute: bool) -> List[dict]:
    """
    在父目录中搜索文件夹
    
    参数:
        parent_dir: 父目录路径
        search_keyword: 搜索关键词
        parent_path: 父路径字符串
        original_query: 原始查询字符串
        is_absolute: 是否为绝对路径
    
    返回:
        文件夹结果列表
    """
    results = []
    try:
        for item in parent_dir.iterdir():
            try:
                # 只处理文件夹
                if not item.is_dir():
                    continue
                
                # 跳过隐藏文件夹
                if item.name.startswith('.'):
                    continue
                
                # 检查是否匹配搜索关键词
                if search_keyword and search_keyword.lower() in item.name.lower():
                    match_score = calculate_match_score(item.name, search_keyword)
                    
                    # 构建返回路径
                    return_path = build_return_path(item, parent_path, original_query, is_absolute)
                    
                    results.append({
                        "name": item.name,
                        "type": "directory",
                        "full_path": return_path,
                        "_score": match_score
                    })
            except (PermissionError, OSError):
                continue
    except (PermissionError, OSError):
        return []
    
    return results


def calculate_match_score(item_name: str, search_keyword: str) -> int:
    """
    计算匹配度分数
    
    参数:
        item_name: 项目名称
        search_keyword: 搜索关键词
    
    返回:
        匹配度分数（越高越相关）
    """
    if not search_keyword:
        return 10
    
    item_name_lower = item_name.lower()
    keyword_lower = search_keyword.lower()
    
    # 精确匹配（最高优先级）
    if item_name_lower == keyword_lower:
        return 100
    
    # 前缀匹配
    if item_name_lower.startswith(keyword_lower):
        return 80
    
    # 包含匹配
    if keyword_lower in item_name_lower:
        return 60
    
    # 部分前缀匹配（前3个字符）
    if len(keyword_lower) >= 3 and item_name_lower.startswith(keyword_lower[:3]):
        return 40
    
    return 10