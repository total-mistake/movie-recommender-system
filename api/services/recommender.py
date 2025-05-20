from typing import List, Dict, Any
from models.hybrid import HybridModel

class RecommenderService:
    def __init__(self):
        self.model = HybridModel()
        
    def get_recommendations(self, user_id: int, top_n: int = 10) -> List[int]:
        recommendations = self.model.predict(user_id, top_n=top_n)
        return [movie_id for movie_id, _ in recommendations]
    
    def get_recent_recommendations(self, user_id: int, top_n: int = 10) -> List[int]:
        recommendations = self.model.content_model.recommend_recent(user_id, top_n=top_n)
        return [movie_id for movie_id, _ in recommendations]
    
    def get_similar_movies(self, movie_id: int, top_n: int = 10) -> List[int]:
        recommendations = self.model.content_model.get_similar_movies(movie_id, top_n=top_n)
        return [movie_id for movie_id, _ in recommendations]

    # Административные методы
    def add_movie(self, movie_dict: Dict[str, Any]) -> None:
        """Добавление нового фильма"""
        self.model.content_model.add_movie(movie_dict=movie_dict)

    def remove_movie(self, movie_id: int) -> None:
        """Удаление фильма"""
        self.model.content_model.remove_movie(movie_id)

    def update_movie(self, movie_data: Dict[str, Any]) -> None:
        """Обновление информации о фильме"""
        self.model.content_model.update_movie(movie_dict=movie_data)

    def retrain_model(self, model_type: str) -> None:
        """Переобучение указанной модели"""
        if model_type == "hybrid":
            self.model.fit()
        elif model_type == "content":
            self.model.content_model.fit()
        elif model_type == "collaborative":
            self.model.collaborative_model.fit()
        else:
            raise ValueError(f"Invalid model type: {model_type}")

    def save_model(self) -> None:
        """Сохранение гибридной модели"""
        self.model._save_model() 