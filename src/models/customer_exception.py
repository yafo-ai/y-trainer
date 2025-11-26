from typing import Any


class CustomerException(Exception):
    def __init__(self, code: int = 400, message: str = "请求错误", data: Any = None):
        self.code = code
        self.message = message
        self.data = data
