from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any, List
from api.auth import verify_admin_token
from api.dependencies import get_recommender_service
from api.services.recommender import RecommenderService
from pydantic import BaseModel
from database.connection import (
    add_movie_by_imdb_id,
    add_movie_to_db,
    update_movie_in_db,
    delete_movie_from_db,
    get_db
)
from sqlalchemy.orm import Session

router = APIRouter(prefix="/admin", tags=["admin"])
security = HTTPBearer()

class ModelRetrainRequest(BaseModel):
    model_type: str  # "hybrid", "content", "collaborative"

class IMDbMovieRequest(BaseModel):
    imdb_id: str

async def verify_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not verify_admin_token(credentials.credentials):
        raise HTTPException(
            status_code=401,
            detail="Invalid admin token"
        )
    return credentials.credentials

@router.post("/movies", dependencies=[Depends(verify_admin)])
async def add_movie(
    movie_data: Dict[str, Any],
    service: RecommenderService = Depends(get_recommender_service)
):
    """Добавление нового фильма"""
    try:
        # Добавляем фильм в БД
        add_movie_to_db(movie_data)
        # Добавляем фильм в модель рекомендаций
        service.add_movie(movie_dict=movie_data)
        return {"message": "Movie added successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/movies/imdb", dependencies=[Depends(verify_admin)])
async def add_movie_by_imdb(
    request: IMDbMovieRequest,
    service: RecommenderService = Depends(get_recommender_service)
):
    """Добавление нового фильма по IMDb ID"""
    try:
        # Получаем данные о фильме и добавляем его в БД
        movie_data = add_movie_by_imdb_id(request.imdb_id)
        if not movie_data:
            raise HTTPException(status_code=404, detail="Movie not found in IMDb")
        
        # Добавляем фильм в модель рекомендаций
        service.add_movie(movie_dict=movie_data)
        return {"message": "Movie added successfully", "movie_data": movie_data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/movies/{movie_id}", dependencies=[Depends(verify_admin)])
async def remove_movie(
    movie_id: int,
    service: RecommenderService = Depends(get_recommender_service)
):
    """Удаление фильма"""
    try:
        # Удаляем фильм из БД
        delete_movie_from_db(movie_id)
        # Удаляем фильм из модели рекомендаций
        service.remove_movie(movie_id)
        return {"message": "Movie removed successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/movies/{movie_id}", dependencies=[Depends(verify_admin)])
async def update_movie(
    movie_id: int,
    movie_data: Dict[str, Any],
    service: RecommenderService = Depends(get_recommender_service)
):
    """Обновление информации о фильме"""
    try:
        # Добавляем ID фильма в данные
        movie_data['Movie_ID'] = movie_id
        # Обновляем фильм в БД
        update_movie_in_db(movie_data)
        # Обновляем фильм в модели рекомендаций
        service.update_movie(movie_data=movie_data)
        return {"message": "Movie updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/models/retrain", dependencies=[Depends(verify_admin)])
async def retrain_model(
    request: ModelRetrainRequest,
    service: RecommenderService = Depends(get_recommender_service)
):
    """Переобучение модели"""
    try:
        service.retrain_model(request.model_type)
        return {"message": f"{request.model_type} model retrained successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/models/save", dependencies=[Depends(verify_admin)])
async def save_model(
    service: RecommenderService = Depends(get_recommender_service)
):
    """Сохранение гибридной модели"""
    try:
        service.save_model()
        return {"message": "Model saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# @router.get("/users", dependencies=[Depends(verify_admin)])
# async def get_users(db: Session = Depends(get_db)):
#     """
#     Получение списка всех пользователей (только для администраторов)
#     """
#     # Здесь будет логика получения списка пользователей
#     pass

# @router.get("/movies", dependencies=[Depends(verify_admin)])
# async def get_all_movies(db: Session = Depends(get_db)):
#     """
#     Получение списка всех фильмов (только для администраторов)
#     """
#     # Здесь будет логика получения списка фильмов
#     pass

# @router.get("/ratings", dependencies=[Depends(verify_admin)])
# async def get_all_ratings(db: Session = Depends(get_db)):
#     """
#     Получение списка всех рейтингов (только для администраторов)
#     """
#     # Здесь будет логика получения списка рейтингов
#     pass 