from fastapi import HTTPException
from api.services.recommender import RecommenderService
from typing import Optional

# Глобальная переменная для хранения сервиса
_recommender_service: Optional[RecommenderService] = None

def set_recommender_service(service: RecommenderService) -> None:
    """
    Установить сервис рекомендаций
    """
    global _recommender_service
    _recommender_service = service

def get_recommender_service() -> RecommenderService:
    """
    Dependency для внедрения сервиса рекомендаций в роуты
    """
    if not _recommender_service:
        raise HTTPException(status_code=503, detail="Сервис рекомендаций не инициализирован")
    return _recommender_service 