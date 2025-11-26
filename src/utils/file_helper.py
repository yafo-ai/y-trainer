import os
from typing import Optional


class FileHelper:

    @staticmethod
    def recursion_paths(root_path, max_depth: Optional[int] = 3):
        """
        获取指定目录下所有文件夹的路径,递归文件夹
        :param root_path: 指定目录
        :param max_depth: 最大递归深度
        :return: 所有文件夹的路径
        """
        paths = {}
        if not os.path.exists(root_path):
            return paths
        if max_depth is None:
            max_depth = float('inf')
        root_path = os.path.abspath(root_path)
        for root, dirs, files in os.walk(root_path):

            current_depth = root[len(root_path):].count(os.sep)
            if max_depth is not None and current_depth >= max_depth:
                dirs[:] = []
            if os.path.basename(root).startswith('.'):
                continue
            for dir_item in dirs:
                if dir_item.startswith('.') or dir_item.startswith('$'):
                    continue
                dir_path = os.path.normpath(os.path.join(root, dir_item))
                dir_path = dir_path + os.sep
                paths[dir_path] = current_depth
        return paths

    @staticmethod
    def get_file_paths(root_path):
        """
        获取指定目录下所有文件夹的路径,不递归文件夹
        :param root_path:
        :return:
        """
        paths = []
        if not os.path.exists(root_path):
            return paths
        for root, dirs, files in os.walk(root_path):
            if os.path.basename(root).startswith('.'):
                continue
            for dir_item in dirs:
                if dir_item.startswith('.') or dir_item.startswith('$'):
                    continue
                tmp_path = os.path.normpath(os.path.join(root, dir_item))
                paths.append(tmp_path + os.sep)
            break
        return paths


