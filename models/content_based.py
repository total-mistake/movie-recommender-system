import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import CONTENT_MODEL_PATH
from .base import BaseModel

class ContentBasedModel(BaseModel):
    def __init__(self, model_path=CONTENT_MODEL_PATH):
        self.model_path = model_path
        self.movie_ids = None
        self.feature_matrix = None
        self.user_profiles = {}
        self.preprocessor = None  # должен быть установлен снаружи или в fit()

        if os.path.exists(self.model_path):
            self.load_model()
        else:
            print(f"[INFO] Контентная модель не найдена по пути {self.model_path}. Нужно вызвать fit().")

    def fit(self, movies_df, ratings_df, preprocessor):
        """
        Строит матрицу признаков фильмов и пользовательские профили.
        """
        self.preprocessor = preprocessor
        self.movie_ids = movies_df['movieId'].values
        self.feature_matrix = self.preprocessor.fit_transform(movies_df)

        # Строим профили пользователей как среднее векторов просмотренных фильмов
        for user_id, group in ratings_df.groupby('userId'):
            watched_ids = group['movieId'].values
            watched_idxs = [np.where(self.movie_ids == mid)[0][0] for mid in watched_ids if mid in self.movie_ids]

            if watched_idxs:
                profile = self.feature_matrix[watched_idxs].mean(axis=0)
                self.user_profiles[user_id] = profile

        self._save_model()

    def _save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({
                'movie_ids': self.movie_ids,
                'feature_matrix': self.feature_matrix,
                'user_profiles': self.user_profiles,
                'preprocessor': self.preprocessor
            }, f)