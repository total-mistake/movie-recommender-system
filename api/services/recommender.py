from typing import List, Dict, Union, Tuple
from models.hybrid import HybridModel

class RecommenderService:
    def __init__(self):
        self.model = HybridModel()
        
    def get_recommendations(self, user_id: int, top_n: int = 10) -> List[int]:
        recommendations = self.model.predict(user_id, top_n=top_n)
        # Извлекаем только ID фильмов из списка кортежей
        return [movie_id for movie_id, _ in recommendations]
    
    def get_recent_recommendations(self, user_id: int, top_n: int = 10) -> List[int]:
        recommendations = self.model.content_model.recommend_recent(user_id, top_n=top_n)
        return [movie_id for movie_id, _ in recommendations]
    
    def get_similar_movies(self, movie_id: int, top_n: int = 10) -> List[int]:
        similar_movies = self.model.content_model.get_similar_movies(movie_id, top_n=top_n)
        return [movie_id for movie_id, _ in similar_movies] 