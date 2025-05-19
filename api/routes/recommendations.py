from fastapi import APIRouter, HTTPException, Depends
from api.schemas.recommendations import RecommendationRequest, MovieIdRequest, RecommendationResponse
from api.services.recommender import RecommenderService
from api.dependencies import get_recommender_service

router = APIRouter()

@router.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    recommender_service: RecommenderService = Depends(get_recommender_service)
):
    """
    Получить рекомендации для пользователя
    """
    try:
        movie_ids = recommender_service.get_recommendations(
            user_id=request.user_id,
            top_n=request.top_n
        )
        return RecommendationResponse(movie_ids=movie_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommendations/recent", response_model=RecommendationResponse)
async def get_recent_recommendations(
    request: RecommendationRequest,
    recommender_service: RecommenderService = Depends(get_recommender_service)
):
    """
    Получить рекомендации на основе недавних просмотров пользователя
    """
    try:
        movie_ids = recommender_service.get_recent_recommendations(
            user_id=request.user_id,
            top_n=request.top_n
        )
        return RecommendationResponse(movie_ids=movie_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommendations/similar", response_model=RecommendationResponse)
async def get_similar_movies(
    request: MovieIdRequest,
    recommender_service: RecommenderService = Depends(get_recommender_service)
):
    """
    Получить похожие фильмы
    """
    try:
        movie_ids = recommender_service.get_similar_movies(
            movie_id=request.movie_id,
            top_n=request.top_n
        )
        return RecommendationResponse(movie_ids=movie_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 