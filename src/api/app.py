import asyncio
from contextlib import asynccontextmanager
import json

from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse
from fastapi_offline import FastAPIOffline
from starlette.responses import JSONResponse, FileResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api import base_api,feature_api, path_api, train_api
from src.models.customer_exception import CustomerException
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 在应用启动时执行
    print("Application startup...")
    # 获取当前运行的事件循环（即主线程的循环）并设置给 collector
    loop = asyncio.get_running_loop()
    from src.api.custom_output import terminal_collector
    terminal_collector.set_event_loop(loop)
    yield
    # 在应用关闭时执行
    print("Application shutdown...")

app = FastAPIOffline(lifespan=lifespan)

app.include_router(feature_api.router)
app.include_router(base_api.router)
app.include_router(train_api.router)
app.include_router(path_api.router)

# app.mount("/", StaticFiles(directory="./frontend/static",html=True), name="static")

@app.middleware("http")
async def wrap_response(request: Request, call_next):
    try:
        response = await call_next(request)
        if not request.url.path.startswith('/api/'):
            return response
        # 过滤导出
        if 'application/vnd.openxmlformats' in response.headers.get('content-type', ''):
            return response
        # 过滤下载
        if 'application/octet-stream' in response.headers.get('content-type', ''):
            return response
        if response.status_code == 200 and "application/json" in response.headers.get('content-type', ''):
            # 获取原始响应体
            raw_body = ''
            async for chunk in response.body_iterator:
                raw_body += chunk.decode('utf-8')
            if 'code' in raw_body and 'data' in raw_body and 'message' in raw_body:
                return JSONResponse(content=json.loads(raw_body), status_code=200)
            wrapped_data = {'code': 200, 'message': 'success', 'data': json.loads(raw_body)}
            return JSONResponse(content=wrapped_data, status_code=200)
        return response
    except Exception as e:
        return JSONResponse(content={'code': 500, 'message': str(e), 'data': None}, status_code=500)


@app.exception_handler(CustomerException)
async def custom_exception_handler(request: Request, exc: CustomerException):
    return JSONResponse(
        status_code=200,
        content={
            "code": exc.code,
            "message": exc.message,
            "data": exc.data
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=200,
        content={
            "code": 400,
            "message": "参数错误",
            "data": {"errors": exc.errors()}
        }
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=200,
        content={
            "code": exc.status_code,
            "message": exc.detail,
            "data": None
        }
    )


@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=200,
        content={
            "code": 500,
            "message": str(exc),
            "data": None
        }
    )


@app.get("/{path:path}", summary="vue前端history模式路由", tags=['前端页面路由'])
async def catch_all(path: str):
    file_path = f"./frontend"
    if path.startswith('static/'):
        if path.endswith('.js'):
            return FileResponse(f"{file_path}/{path}", media_type='application/javascript')
        if path.endswith('.css'):
            return FileResponse(f"{file_path}/{path}", media_type='text/css')
    if path.endswith('.svg'):
        return FileResponse(f"{file_path}/{path}", media_type='image/svg+xml')
    if path.endswith('.js'):
        return FileResponse(f"{file_path}/{path}", media_type='application/javascript')
        # 处理字体文件
    if path.endswith('.woff'):
        return FileResponse(f"{file_path}/{path}", media_type='font/woff')
    if path.endswith('.woff2'):
        return FileResponse(f"{file_path}/{path}", media_type='font/woff2')
    if path.endswith('.ttf'):
        return FileResponse(f"{file_path}/{path}", media_type='font/ttf')
    if path.endswith('.eot'):
        return FileResponse(f"{file_path}/{path}", media_type='application/vnd.ms-fontobject')
    return FileResponse(f"{file_path}/index.html")