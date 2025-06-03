from fastapi import HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from typing import Optional
import os
import secrets
from datetime import datetime, timedelta
import jwt
from config import (
    JWT_SECRET_KEY,
    JWT_ALGORITHM,
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES
)

# В реальном приложении эти значения должны храниться в безопасном месте
# и быть загружены из переменных окружения
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", secrets.token_hex(32))

api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)) -> bool:
    """Проверка API ключа администратора"""
    if api_key != ADMIN_API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return True

def create_access_token(user_id: int) -> str:
    """
    Создание JWT токена для пользователя
    """
    expire = datetime.utcnow() + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {
        "sub": str(user_id),
        "exp": expire
    }
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def verify_token(token: str) -> Optional[int]:
    """
    Проверка JWT токена и получение ID пользователя
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id = int(payload.get("sub"))
        return user_id
    except jwt.PyJWTError:
        return None

def get_admin_token() -> str:
    """
    Получение JWT токена для администратора
    """
    # В продакшене здесь должна быть проверка API ключа
    return create_access_token(0)  # 0 - специальный ID для администратора

def verify_admin_token(token: str) -> bool:
    """
    Проверка токена администратора.
    Возвращает True, если токен действителен.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload.get("role") == "admin"
    except jwt.PyJWTError:
        return False 