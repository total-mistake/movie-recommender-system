from .base import BaseModel
from .content_based import ContentBasedModel
from .collaborative import CollaborativeModel

class HybridModel(BaseModel):
    def __init__(self, alpha=0.6):
        self.alpha = alpha  # вес для контентной модели (1 - alpha — для коллаборативной)
        # Инициализируем внутренние модели
        self.content_model = ContentBasedModel()
        self.collaborative_model = CollaborativeModel()
        if self.model_exists():
            self.load_model()
        else:
            print("[INFO] Одна или обе внутренние модели не найдены. Нужно вызвать fit().")

    def model_exists(self):
        """Проверяет существование обеих внутренних моделей"""
        return (self.content_model.model_exists() and 
                self.collaborative_model.model_exists())

    def fit(self, movies_df, ratings_df):
        """
        Обучает внутренние модели.
        """
        self.content_model.fit(movies_df, ratings_df)
        self.collaborative_model.fit(ratings_df)
        self._save_model()

    def _save_model(self):
        self.collaborative_model._save_model()
        self.content_model._save_model()

    def load_model(self):
        if self.model_exists():
            # Загружаем только веса и параметры внутренних моделей
            self.collaborative_model.load_model()
            self.content_model.load_model()

    def predict(self, user_id, top_n=None):
        """
        Объединяет рекомендации контентной и коллаборативной модели по весам.
        Для новых пользователей использует только контентную модель.
        """
        try:
            # Пробуем получить рекомендации от обеих моделей
            content_recs = dict(self.content_model.predict(user_id, top_n=None))
            collab_recs = dict(self.collaborative_model.predict(user_id, top_n=None))
            
            # Нормализуем scores от коллаборативной модели в [0, 1]
            collab_recs = {
                mid: (score - 0.5) / 4.5
                for mid, score in collab_recs.items()
            }
            
            all_movie_ids = set(content_recs.keys()) | set(collab_recs.keys())
            combined_scores = {}
            for mid in all_movie_ids:
                content_score = content_recs.get(mid, 0)
                collab_score = collab_recs.get(mid, 0)
                score = self.alpha * content_score + (1 - self.alpha) * collab_score
                if score > 0:  # Добавляем только фильмы с положительным скором
                    combined_scores[mid] = score
                    
            # Сортировка и top_n
            sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_recs[:top_n]
            
        except ValueError as e:
            if "отсутствует в обучающей выборке" in str(e):
                # Если пользователь новый, используем только контентную модель
                print(f"[INFO] Пользователь {user_id} новый, используем только контентную модель")
                content_recs = dict(self.content_model.predict(user_id, top_n=None))
                sorted_recs = sorted(content_recs.items(), key=lambda x: x[1], reverse=True)
                return sorted_recs[:top_n]
            raise
