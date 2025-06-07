from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
from enum import Enum
from api.schemas.movie import MovieResponse, MovieListResponse
from api.schemas.recommendations import MovieIdRequest, RecommendationResponse
from database.connection import get_movies_data, get_movie_by_id, get_genres
from api.services.recommender import RecommenderService
from api.dependencies import get_recommender_service
from fastapi_cache.decorator import cache
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
import numpy as np
import logging
import pandas as pd
from statistics import mean

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["movies"])

class SortDirection(str, Enum):
    """Направление сортировки"""
    ASC = "asc"
    DESC = "desc"

class SortField(str, Enum):
    """Поля, по которым можно сортировать фильмы"""
    TITLE = "title"
    YEAR = "year"
    RATING = "rating"
    RATING_COUNT = "rating_count"
    POPULARITY = "popularity"

# Константы для расчета популярности
C = 1000  # Минимальное количество голосов для учета
M = 3.0   # Средний рейтинг по всем фильмам

def calculate_popularity_score(rating: float, vote_count: int) -> float:
    """
    Расчет популярности фильма по формуле Байеса:
    (v * R + m * M) / (v + m)
    где:
    v - количество голосов
    R - рейтинг фильма
    m - минимальное количество голосов (C)
    M - средний рейтинг по всем фильмам
    """
    if rating is None or vote_count is None or vote_count == 0:
        return 0.0
    
    return (vote_count * rating + C * M) / (vote_count + C)

def convert_numpy_types(obj: Any) -> Any:
    """Преобразование numpy типов в Python типы"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def prepare_movies_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Подготовка данных фильмов для кеширования"""
    # Заменяем NaN на None
    df = df.replace({np.nan: None})
    
    # Рассчитываем популярность для всех фильмов сразу
    df['Popularity'] = df.apply(
        lambda row: calculate_popularity_score(row['Rating'], row['Rating_Count']),
        axis=1
    )
    
    # Конвертируем DataFrame в список словарей
    movies_list = []
    for _, row in df.iterrows():
        movie_dict = row.to_dict()
        converted_dict = convert_numpy_types(movie_dict)
        movies_list.append(converted_dict)
    
    return movies_list

def sort_movies(movies: List[Dict[str, Any]], sort_by: SortField, sort_dir: SortDirection) -> List[Dict[str, Any]]:
    """Сортировка списка фильмов"""
    # Маппинг полей для сортировки
    field_mapping = {
        SortField.TITLE: "Title",
        SortField.YEAR: "Year",
        SortField.RATING: "Rating",
        SortField.RATING_COUNT: "Rating_Count",
        SortField.POPULARITY: "Popularity"  # Добавляем поле популярности
    }
    
    sort_key = field_mapping[sort_by]
    reverse = sort_dir == SortDirection.DESC
    
    # Специальная обработка для числовых полей
    if sort_by in [SortField.YEAR, SortField.RATING, SortField.RATING_COUNT, SortField.POPULARITY]:
        return sorted(
            movies,
            key=lambda x: float(x[sort_key]) if x[sort_key] is not None else float('-inf'),
            reverse=reverse
        )
    
    # Для текстовых полей
    return sorted(
        movies,
        key=lambda x: str(x[sort_key]).lower() if x[sort_key] is not None else "",
        reverse=reverse
    )

@router.get("/genres", response_model=List[str])
async def get_all_genres():
    """Получить список всех доступных жанров"""
    try:
        genres = get_genres()
        return [genre["Genre"] for genre in genres]  # Возвращаем только названия жанров
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=MovieListResponse)
@cache(expire=3600, namespace="movies_list")
async def get_movies(
    page: int = Query(1, ge=1, description="Номер страницы"),
    page_size: int = Query(20, ge=1, le=100, description="Количество элементов на странице"),
    sort_by: Optional[SortField] = Query(
        None,
        description="Поле для сортировки (title, year, rating, rating_count, popularity)"
    ),
    sort_dir: Optional[SortDirection] = Query(
        None,
        description="Направление сортировки (asc, desc)"
    )
):
    """
    Получить постраничный список фильмов с возможностью сортировки.
    
    Параметры сортировки:
    - sort_by: поле для сортировки (title, year, rating, rating_count, popularity)
    - sort_dir: направление сортировки (asc, desc)
    
    Поле popularity рассчитывается с учетом рейтинга и количества отзывов
    по формуле Байеса для более точной оценки популярности.
    
    Если параметры сортировки не указаны, возвращается список в исходном порядке.
    """
    try:
        logger.info("Получение данных о фильмах из кеша")
        movies_list = await get_cached_movies()
        
        logger.info(f"Получено {len(movies_list)} фильмов")
        
        # Применяем сортировку, если указаны параметры
        if sort_by is not None and sort_dir is not None:
            logger.info(f"Сортировка по полю {sort_by} в направлении {sort_dir}")
            movies_list = sort_movies(movies_list, sort_by, sort_dir)
        
        # Вычисляем пагинацию
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        # Получаем фильмы для текущей страницы
        page_movies = movies_list[start_idx:end_idx]
        
        total = len(movies_list)
        total_pages = (total + page_size - 1) // page_size
        
        logger.info(f"Возвращаем {len(page_movies)} фильмов для страницы {page}")
        
        return {
            "movies": page_movies,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages
        }
    except Exception as e:
        logger.error(f"Ошибка при получении списка фильмов: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@cache(expire=3600, namespace="movies_data")
async def get_cached_movies() -> List[Dict[str, Any]]:
    """Получение кешированного списка всех фильмов"""
    logger.info("Загрузка данных о фильмах из базы данных")
    movies_df = get_movies_data()
    logger.info(f"Загружено {len(movies_df)} фильмов из базы данных")
    
    # Преобразуем DataFrame в список словарей перед кешированием
    movies_list = prepare_movies_data(movies_df)
    logger.info("Данные подготовлены для кеширования")
    
    return movies_list

@router.get("/{movie_id}", response_model=MovieResponse)
async def get_movie(movie_id: int):
    """Получить детальную информацию о фильме по его ID"""
    try:
        movie = get_movie_by_id(movie_id)
        if not movie:
            raise HTTPException(status_code=404, detail="Фильм не найден")
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
    """Получить похожие фильмы на основе ID фильма"""
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