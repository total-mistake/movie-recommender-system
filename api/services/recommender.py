from typing import List, Dict, Any
from models.hybrid import HybridModel
from database.connection import get_movies_data, get_ratings_data, add_movie_to_db
import logging

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class RecommenderService:
    def __init__(self):
        self.model = HybridModel()
        
    def get_recommendations(self, user_id: int, top_n: int = 10) -> List[int]:
        """Get personalized movie recommendations for a user"""
        recommendations = self.model.predict(user_id, top_n=top_n)
        # recommendations уже список кортежей (id, score)
        return [movie_id for movie_id, _ in recommendations]
    
    def get_recent_recommendations(self, user_id: int, top_n: int = 10) -> List[int]:
        """Get recommendations based on recent user activity"""
        recommendations = self.model.content_model.recommend_recent(user_id, top_n=top_n)
        # recommendations уже список кортежей (id, score)
        return [movie_id for movie_id, _ in recommendations]
    
    def get_similar_movies(self, movie_id: int, top_n: int = 10) -> List[int]:
        """Get similar movies based on movie ID"""
        recommendations = self.model.content_model.get_similar_movies(movie_id, top_n=top_n)
        # recommendations уже список кортежей (id, score)
        return [movie_id for movie_id, _ in recommendations]
    
    def add_rating(self, user_id: int, movie_id: int, rating: float) -> None:
        self.model.content_model.update_user_profile(user_id, movie_id, rating)

    def add_new_user(self, user_id: int, favorite_genres: str) -> None:
        """Добавление нового пользователя"""
        # Логируем входные жанры
        logger.debug(f"RecommenderService.add_new_user: Input genres: {favorite_genres}")
        
        # Разбиваем жанры
        genres_list = [genre.strip() for genre in favorite_genres.split(',') if genre.strip()]
        logger.debug(f"RecommenderService.add_new_user: Split genres: {genres_list}")
        
        # Добавляем пользователя
        self.model.content_model.add_new_user(user_id, genres_list)
        
        # Логируем профиль пользователя после добавления
        user_profile = self.model.content_model.get_user_profile(user_id)
        if user_profile is not None:
            logger.debug(f"RecommenderService.add_new_user: User profile shape: {user_profile.shape}")
            # Получаем ненулевые компоненты профиля
            non_zero = user_profile.nonzero()
            if len(non_zero[0]) > 0:
                logger.debug(f"RecommenderService.add_new_user: Non-zero components in profile: {non_zero[0][:5]}")

    # Административные методы
    def add_movie(self, movie_dict: Dict[str, Any]) -> None:
        """Добавление нового фильма"""
        # Добавляем фильм в БД
        add_movie_to_db(movie_dict)
        # Добавляем фильм в модель
        self.model.content_model.add_movie(movie_dict=movie_dict)

    def remove_movie(self, movie_id: int) -> None:
        """Удаление фильма"""
        self.model.content_model.remove_movie(movie_id)

    def update_movie(self, movie_data: Dict[str, Any]) -> None:
        """Обновление информации о фильме"""
        self.model.content_model.update_movie(movie_dict=movie_data)

    def retrain_model(self, model_type: str) -> None:
        """Переобучение указанной модели"""
        # Получаем данные из БД
        movies = get_movies_data()
        ratings = get_ratings_data()
        
        if model_type == "hybrid":
            self.model.fit(movies, ratings)
        elif model_type == "content":
            self.model.content_model.fit(movies, ratings)
        elif model_type == "collaborative":
            self.model.collaborative_model.fit(ratings)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

    def save_model(self) -> None:
        """Сохранение гибридной модели"""
        self.model._save_model() 