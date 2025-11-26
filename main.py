import uvicorn

from src.configs.server_config import FASTAPI_HOST, FASTAPI_PORT

if __name__ == '__main__':
    uvicorn.run(app="src.api.app:app", host=FASTAPI_HOST, port=FASTAPI_PORT)
