from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.routes import recommendations, health, admin
from api.services.recommender import RecommenderService
from api.dependencies import set_recommender_service
from api.auth import get_admin_token
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Movie Recommender API",
    description="API для получения рекомендаций фильмов",
    version="1.0.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене заменить на конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Флаг готовности сервера
is_ready = False

@app.on_event("startup")
async def startup_event():
    """
    Инициализация модели при запуске сервера
    """
    global is_ready
    try:
        logger.info("Загрузка модели рекомендаций...")
        # Инициализация сервиса рекомендаций
        service = RecommenderService()
        set_recommender_service(service)
        is_ready = True
        logger.info("Модель успешно загружена")
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {str(e)}")
        raise

@app.middleware("http")
async def check_ready_middleware(request, call_next):
    """
    Middleware для проверки готовности сервера
    """
    if not is_ready and request.url.path != "/api/health":
        raise HTTPException(status_code=503, detail="Сервер не готов к обработке запросов")
    return await call_next(request)

# Подключаем роуты
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(recommendations.router, prefix="/api", tags=["recommendations"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])

# Роут для получения токена администратора
@app.post("/api/admin/token", tags=["admin"])
async def get_token():
    """
    Получение JWT токена для администратора
    Требует валидный API ключ в заголовке X-API-Key
    """
    return {"token": get_admin_token()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 