
import asyncio
import logging
import sys
from typing import List
from fastapi import WebSocket


class WebSocketCollector:
    def __init__(self):
        self.output: List[str] = []
        self.websockets: List[WebSocket] = []
        # 我们需要一种方式来获取主线程的事件循环
        # 最简单的方法是在应用启动时设置它
        self.loop = None 

    def set_event_loop(self, loop):
        """在应用启动时调用，用于设置主事件循环的引用"""
        self.loop = loop

    async def add_ws(self, websocket: WebSocket):
        await websocket.accept()
        self.websockets.append(websocket)
        # 发送最近的100行历史记录给新连接的客户端
        for line in self.output[-100:]:
            await websocket.send_text(line)

    def remove_ws(self, websocket: WebSocket):
        if websocket in self.websockets:
            self.websockets.remove(websocket)

    def write(self, text: str):
        """这个方法现在可以从任何线程（包括同步训练线程）安全调用"""
        self.output.append(text)
        if len(self.output) > 1000:
            self.output.pop(0)
        
        # 关键修复：使用 run_coroutine_threadsafe
        if self.loop:
            # 将 broadcast 协程安全地提交到主事件循环
            asyncio.run_coroutine_threadsafe(self.broadcast(text), self.loop)
        else:
            # 如果循环还未设置（例如在应用完全启动前），可以选择打印到标准错误
            # 或者直接忽略，取决于你的需求
            sys.stderr.write(f"WebSocketCollector: Event loop not set. Cannot broadcast: {text}\n")

    async def broadcast(self, text: str):
        """这个方法始终在主事件循环中执行，因此可以安全地使用 await"""
        disconnected = []
        for ws in self.websockets:
            try:
                await ws.send_text(text)
            except Exception as e:
                # 可以记录一下断开连接的错误
                print(f"WebSocket disconnected or error: {e}")
                disconnected.append(ws)
        
        for ws in disconnected:
            self.remove_ws(ws)


terminal_collector = WebSocketCollector()


class RedirectOutput:
    def __init__(self, collector):
        self.collector = collector

    def write(self, text):
        self.collector.write(text)
        sys.__stdout__.write(text)

    def flush(self):
        sys.__stdout__.flush()


class RedirectError:
    def __init__(self, collector):
        self.collector = collector

    def write(self, text):
        self.collector.write(text)
        sys.__stderr__.write(text)

    def flush(self):
        sys.__stderr__.flush()


# 应用重定向
sys.stdout = RedirectOutput(terminal_collector)
sys.stderr = RedirectError(terminal_collector)


class LoggingHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        terminal_collector.write(log_entry + '\n')


# 配置root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(LoggingHandler())

# 确保uvicorn的日志也被捕获
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.setLevel(logging.INFO)
uvicorn_logger.addHandler(LoggingHandler())

uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.setLevel(logging.INFO)
uvicorn_access_logger.addHandler(LoggingHandler())