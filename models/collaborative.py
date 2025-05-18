import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from .base import BaseModel
from surprise import SVD, Dataset, Reader
from config import COLLABORATIVE_MODEL_PATH, SVD_PARAMS

class CollaborativeModel(BaseModel):
    def __init__(self, model_path=COLLABORATIVE_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.all_movie_ids = []
        self.user_rated_movies = {}

        if os.path.exists(self.model_path):
            self.load_model()
        else:
            print(f"[INFO] Модель не найдена по пути {self.model_path}. Нужно вызвать fit().")

    def fit(self, ratings_df):
        """
        Обучает SVD и сохраняет модель.
        """
        reader = Reader(rating_scale=(ratings_df.rating.min(), ratings_df.rating.max()))
        data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
        self.trainset = data.build_full_trainset()

        self.model = SVD(**SVD_PARAMS, random_state=9)
        self.model.fit(self.trainset)

        # Собираем все уникальные фильмы и просмотренные пользователями
        self.all_movie_ids = ratings_df["movieId"].unique().tolist()

        self.user_rated_movies = defaultdict(set)
        for _, row in ratings_df.iterrows():
            self.user_rated_movies[int(row.userId)].add(int(row.movieId))

        # Сохраняем ratings_df отдельно для восстановления trainset
        ratings_path = self.model_path.replace(".pkl", "_ratings.parquet")
        ratings_df.to_parquet(ratings_path, index=False)

        self._save_model()

    def _save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "all_movie_ids": self.all_movie_ids,
                "user_rated_movies": self.user_rated_movies
            }, f)

    def load_model(self):
        with open(self.model_path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.all_movie_ids = data["all_movie_ids"]
            self.user_rated_movies = data["user_rated_movies"]
    
    def predict(self, user_id):
        if self.model is None:
            raise RuntimeError("Модель не загружена и не обучена.")

        known_ids = self.user_rated_movies.get(user_id, set())
        candidates = [mid for mid in self.all_movie_ids if mid not in known_ids]

        predictions = [
            (movie_id, self.model.predict(user_id, movie_id).est)
            for movie_id in candidates
        ]
        return predictions

    def recommend(self, user_id, n=10):
        predictions = self.predict(user_id)
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]