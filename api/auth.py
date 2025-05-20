from fastapi import HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from typing import Optional
import os
import secrets
from datetime import datetime, timedelta
import jwt

# В реальном приложении эти значения должны храниться в безопасном месте
# и быть загружены из переменных окружения
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", secrets.token_hex(32))
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION = timedelta(hours=24)

api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)) -> bool:
    """Проверка API ключа администратора"""
    if api_key != ADMIN_API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return True

def create_admin_token() -> str:
    """Создание JWT токена для администратора"""
    expiration = datetime.utcnow() + JWT_EXPIRATION
    token_data = {
        "exp": expiration,
        "role": "admin"
    }
    return jwt.encode(token_data, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_admin_token(token: str) -> bool:
    """Проверка JWT токена администратора"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Invalid token role")
        return True
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=403, detail="Invalid token")

def get_admin_token(api_key: bool = Depends(verify_api_key)) -> str:
    """Получение JWT токена администратора"""
    return create_admin_token() 