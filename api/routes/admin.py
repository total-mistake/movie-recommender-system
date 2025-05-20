from fastapi import APIRouter, Depends, HTTPException, Header
from typing import Dict, Any
from api.auth import verify_admin_token
from api.dependencies import get_recommender_service
from api.services.recommender import RecommenderService
from pydantic import BaseModel

router = APIRouter()

class ModelRetrainRequest(BaseModel):
    model_type: str  # "hybrid", "content", "collaborative"

def verify_admin_auth(authorization: str = Header(...)) -> bool:
    """Проверка авторизации администратора"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.split(" ")[1]
    return verify_admin_token(token)

@router.post("/movies", dependencies=[Depends(verify_admin_auth)])
async def add_movie(
    movie_data: Dict[str, Any],
    service: RecommenderService = Depends(get_recommender_service)
):
    """Добавление нового фильма"""
    try:
        service.add_movie(movie_dict=movie_data)
        return {"message": "Movie added successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/movies/{movie_id}", dependencies=[Depends(verify_admin_auth)])
async def remove_movie(
    movie_id: int,
    service: RecommenderService = Depends(get_recommender_service)
):
    """Удаление фильма"""
    try:
        service.remove_movie(movie_id)
        return {"message": "Movie removed successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/movies/{movie_id}", dependencies=[Depends(verify_admin_auth)])
async def update_movie(
    movie_id: int,
    movie_data: Dict[str, Any],
    service: RecommenderService = Depends(get_recommender_service)
):
    """Обновление информации о фильме"""
    try:
        service.update_movie(movie_data=movie_data)
        return {"message": "Movie updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/models/retrain", dependencies=[Depends(verify_admin_auth)])
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

@router.post("/models/save", dependencies=[Depends(verify_admin_auth)])
async def save_model(
    service: RecommenderService = Depends(get_recommender_service)
):
    """Сохранение гибридной модели"""
    try:
        service.save_model()
        return {"message": "Model saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 