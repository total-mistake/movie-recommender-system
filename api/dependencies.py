from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from api.services.recommender import RecommenderService
from api.auth import verify_token
from typing import Optional, Dict

# Глобальная переменная для хранения сервиса
_recommender_service: Optional[RecommenderService] = None

security = HTTPBearer()

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

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[Dict]:
    """
    Получение текущего пользователя из токена.
    Возвращает None, если пользователь не аутентифицирован.
    """
    user_id = verify_token(credentials.credentials)
    if user_id is None:
        return None
    return {"user_id": user_id} 