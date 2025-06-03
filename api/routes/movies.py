from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any
from api.schemas.movie import MovieResponse, MovieListResponse
from api.schemas.recommendations import MovieIdRequest, RecommendationResponse
from database.connection import get_movies_data, get_movie_by_id, get_genres
from api.services.recommender import RecommenderService
from api.dependencies import get_recommender_service
from fastapi_cache.decorator import cache
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend

router = APIRouter(tags=["movies"])

@router.get("/genres", response_model=List[str])
@cache(expire=3600)  # Кешируем список жанров на 1 час
async def get_all_genres():
    """Get all available genres"""
    try:
        genres = get_genres()
        return [genre["Genre"] for genre in genres]  # Возвращаем только названия жанров
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=MovieListResponse)
@cache(expire=3600, namespace="movies_list")  # Кешируем на 1 час с namespace
async def get_movies(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Number of items per page")
):
    """
    Get a paginated list of movies.
    Results are cached for 1 hour to improve performance.
    Cache is shared between all pagination parameters.
    """
    try:
        # Получаем все фильмы (это будет кешироваться отдельно)
        movies_df = await get_cached_movies()
        
        # Calculate pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        # Get paginated movies and convert to list of dicts
        paginated_movies = movies_df.iloc[start_idx:end_idx].to_dict('records')
        total = len(movies_df)
        total_pages = (total + page_size - 1) // page_size
        
        return {
            "movies": paginated_movies,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@cache(expire=3600, namespace="movies_data")  # Кешируем полный список фильмов на 1 час
async def get_cached_movies():
    """Получение кешированного списка всех фильмов"""
    return get_movies_data()

@router.get("/{movie_id}", response_model=MovieResponse)
async def get_movie(movie_id: int):
    """Get movie details by ID"""
    try:
        movie = get_movie_by_id(movie_id)
        if not movie:
            raise HTTPException(status_code=404, detail="Movie not found")
        return movie
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/similar", response_model=RecommendationResponse)
async def get_similar_movies(
    request: MovieIdRequest,
    recommender_service: RecommenderService = Depends(get_recommender_service)
):
    """Get similar movies based on movie ID"""
    try:
        similar_movies = recommender_service.get_similar_movies(
            movie_id=request.movie_id,
            top_n=request.top_n
        )
        return RecommendationResponse(movie_ids=similar_movies)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 