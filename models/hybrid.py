import os
import pickle
import time
from .base import BaseModel
from .content_based import ContentBasedModel
from .collaborative import CollaborativeModel
from config import HYBRID_MODEL_PATH

class HybridModel(BaseModel):
    def __init__(self, alpha=0.1, model_path=HYBRID_MODEL_PATH):
        self.alpha = alpha  # вес для контентной модели (1 - alpha — для коллаборативной)
        self.model_path = model_path
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            print(f"[INFO] Модель не найдена по пути {self.model_path}. Нужно вызвать fit().")

    def fit(self, movies_df, ratings_df):
        """
        Обучает недообученные модели. Если модель уже обучена — пропускает обучение.
        """
        # Проверка, обучена ли контентная модель
        if not self.content_model.user_profiles:
            print("[INFO] Контентная модель не обучена. Запуск обучения...")
            self.content_model.fit(movies_df, ratings_df)

        # Проверка, обучена ли коллаборативная модель
        if self.collaborative_model.model is None:
            print("[INFO] Коллаборативная модель не обучена. Запуск обучения...")
            self.collaborative_model.fit(ratings_df)

        # Гибридной модели отдельное обучение не требуется — только весовая настройка
        self._save_model()

    def _save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "alpha": self.alpha
            }, f)
        self.collaborative_model._save_model()
        self.content_model._save_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
                self.alpha = data["alpha"]
        # Также загружаем внутренние модели
        self.collaborative_model = CollaborativeModel()
        self.content_model = ContentBasedModel()
        self.content_model.predict(1)

    def predict(self, user_id, top_n=10):
        """
        Объединяет рекомендации контентной и коллаборативной модели по весам.
        """
        start = time.time()
        print(f"[INFO] Генерация рекомендаций для пользователя {user_id}...")
        content_recs = dict(self.content_model.predict(user_id, top_n=None))
        print(f"[INFO] Получено {len(content_recs)} рекомендаций от контентной модели. {time.time() - start:.2f} секунд")
        start = time.time()
        collab_recs = dict(self.collaborative_model.predict(user_id, top_n=None))
        print(f"[INFO] Получено {len(collab_recs)} рекомендаций от коллаборативной модели. {time.time() - start:.2f} секунд")
        start = time.time()
        all_movie_ids = set(content_recs.keys()) | set(collab_recs.keys())
        combined_scores = {}
        start = time.time()
        for mid in all_movie_ids:
            content_score = content_recs.get(mid, 0)
            collab_score = collab_recs.get(mid, 0)
            score = self.alpha * content_score + (1 - self.alpha) * collab_score
            combined_scores[mid] = score
        # Сортировка и top_n
        sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:top_n]
