from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from api.routes import admin, users, movies
from api.services.recommender import RecommenderService
from api.dependencies import set_recommender_service
from api.auth import get_admin_token
import logging
from config import (
    API_TITLE, API_DESCRIPTION, API_VERSION,
    API_HOST, API_PORT,
    CORS_ORIGINS, CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_METHODS, CORS_ALLOW_HEADERS
)
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Инициализация при запуске и очистка при завершении
    """
    try:
        # Инициализация in-memory кеша
        FastAPICache.init(InMemoryBackend(), prefix="movie-cache")
        logger.info("In-memory cache initialized")
        
        # Инициализация модели рекомендаций
        logger.info("Загрузка модели рекомендаций...")
        service = RecommenderService()
        set_recommender_service(service)
        app.state.is_ready = True
        logger.info("Модель успешно загружена")
        
        yield
        
    except Exception as e:
        logger.error(f"Ошибка при инициализации: {str(e)}")
        raise

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)

# Инициализация состояния приложения
app.state.is_ready = False

@app.on_event("startup")
async def startup_event():
    """
    Инициализация модели при запуске сервера
    """
    try:
        logger.info("Загрузка модели рекомендаций...")
        service = RecommenderService()
        set_recommender_service(service)
        app.state.is_ready = True
        logger.info("Модель успешно загружена")
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {str(e)}")
        raise

@app.middleware("http")
async def check_ready_middleware(request, call_next):
    """
    Middleware для проверки готовности сервера
    """
    if not app.state.is_ready and request.url.path != "/api/health":
        raise HTTPException(status_code=503, detail="Сервер не готов к обработке запросов")
    return await call_next(request)

# Подключаем роуты
app.include_router(
    admin.router,
    prefix="/admin",
    tags=["admin"]
)

app.include_router(
    users.router,
    prefix="/users",
    tags=["users"]
)

app.include_router(
    movies.router,
    prefix="/movies",
    tags=["movies"]
)

# Роут для получения токена администратора
@app.post("/admin/token", tags=["admin"])
async def get_token():
    """
    Получение JWT токена для администратора
    Требует валидный API ключ в заголовке X-API-Key
    """
    return {"token": get_admin_token()}

@app.get("/health")
async def health_check():
    """Проверка работоспособности API"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT) 