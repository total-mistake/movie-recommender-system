from fastapi import APIRouter, Depends, HTTPException, Header, Query
from pydantic import BaseModel
from typing import Optional, List
from database.connection import add_user_to_db, verify_user_credentials, add_rating_to_db
from api.auth import create_access_token, verify_token
from api.services.recommender import RecommenderService
from api.dependencies import get_recommender_service, get_current_user
from api.schemas.recommendations import RecommendationRequest, RecommendationResponse

router = APIRouter()

class UserCreate(BaseModel):
    username: str
    password: str
    favorite_genres: str

class UserLogin(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int

class RatingRequest(BaseModel):
    movie_id: int
    rating: float

@router.post("/register", response_model=TokenResponse)
async def register_user(user_data: UserCreate, recommender_service: RecommenderService = Depends(get_recommender_service)):
    """
    Регистрация нового пользователя.
    Возвращает JWT токен для аутентификации.
    """
    try:
        user_id = add_user_to_db(user_data.username, user_data.password)
        access_token = create_access_token(user_id)
        recommender_service.add_new_user(user_id, user_data.favorite_genres)
        return TokenResponse(
            access_token=access_token,
            user_id=user_id
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/login", response_model=TokenResponse)
async def login_user(user_data: UserLogin):
    """
    Аутентификация пользователя.
    Возвращает JWT токен при успешной аутентификации.
    """
    user_id = verify_user_credentials(user_data.username, user_data.password)
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Неверное имя пользователя или пароль"
        )
    
    access_token = create_access_token(user_id)
    return TokenResponse(
        access_token=access_token,
        user_id=user_id
    )

@router.get("/me")
async def get_current_user(authorization: str = Header(...)):
    """
    Получение информации о текущем пользователе.
    Требует JWT токен в заголовке Authorization.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    token = authorization.split(" ")[1]
    user_id = verify_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return {"user_id": user_id}

@router.post("/ratings")
async def add_rating(
    rating_data: RatingRequest,
    current_user: dict = Depends(get_current_user),
    recommender_service: RecommenderService = Depends(get_recommender_service)
):
    """
    Добавление рейтинга пользователя.
    Требует JWT токен в заголовке Authorization.
    """
    try:
        user_id = current_user["user_id"]
        # Добавляем рейтинг в БД
        add_rating_to_db(
            user_id=user_id,
            movie_id=rating_data.movie_id,
            rating=rating_data.rating
        )
        
        # Обновляем профиль пользователя в контентной модели
        recommender_service.add_rating(user_id, rating_data.movie_id, rating_data.rating)
        
        return {"message": "Rating added successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/recommendations", response_model=RecommendationResponse)
async def get_user_recommendations(
    top_n: int = Query(10, ge=1, le=100),
    current_user: dict = Depends(get_current_user),
    recommender_service: RecommenderService = Depends(get_recommender_service)
):
    """
    Получить персонализированные рекомендации для авторизованного пользователя
    """
    try:
        user_id = current_user["user_id"]
        movie_ids = recommender_service.get_recommendations(
            user_id=user_id,
            top_n=top_n
        )
        return RecommendationResponse(movie_ids=movie_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendations/recent", response_model=RecommendationResponse)
async def get_user_recent_recommendations(
    top_n: int = Query(10, ge=1, le=100),
    current_user: dict = Depends(get_current_user),
    recommender_service: RecommenderService = Depends(get_recommender_service)
):
    """
    Получить рекомендации на основе недавних просмотров авторизованного пользователя
    """
    try:
        user_id = current_user["user_id"]
        movie_ids = recommender_service.get_recent_recommendations(
            user_id=user_id,
            top_n=top_n
        )
        return RecommendationResponse(movie_ids=movie_ids)
    except ValueError as e:
        if "отсутствует в обучающей выборке" in str(e):
            # Для новых пользователей возвращаем пустой список
            return RecommendationResponse(movie_ids=[])
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 