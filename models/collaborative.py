import os
import pickle
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

        self.model = SVD(**SVD_PARAMS)
        self.model.fit(self.trainset)

        # Собираем все уникальные фильмы и просмотренные пользователями
        self.all_movie_ids = ratings_df["movieId"].unique().tolist()

        self.user_rated_movies = defaultdict(set)
        for _, row in ratings_df.iterrows():
            self.user_rated_movies[int(row.userId)].add(int(row.movieId))

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
    
    def predict(self, user_id, top_n=None):
        if self.model is None:
            raise RuntimeError("Модель не загружена и не обучена.")

        # Получаем внутренние индексы пользователя и фильмов в trainset
        trainset = self.model.trainset
        try:
            uid_inner = trainset.to_inner_uid(user_id)
        except ValueError:
            raise ValueError(f"Пользователь {user_id} отсутствует в обучающей выборке")

        known_ids = self.user_rated_movies.get(user_id, set())
        candidates = [mid for mid in self.all_movie_ids if mid not in known_ids]

        # Преобразуем внешние id фильмов во внутренние
        candidate_iids = []
        valid_candidate_ids = []
        for mid in candidates:
            try:
                iid_inner = trainset.to_inner_iid(mid)
                candidate_iids.append(iid_inner)
                valid_candidate_ids.append(mid)
            except ValueError:
                # Если фильм не в trainset, пропускаем
                continue

        # Получаем матрицы
        pu = self.model.pu[uid_inner]           # вектор пользователя (k,)
        qi = self.model.qi[candidate_iids]      # матрица кандидатов (len x k)
        bu = self.model.bu[uid_inner]            # смещение пользователя
        bi = self.model.bi[candidate_iids]       # смещения фильмов
        global_mean = self.model.trainset.global_mean

        # Вектор предсказаний (numpy)
        preds = global_mean + bu + bi + qi.dot(pu)

        predictions = list(zip(valid_candidate_ids, preds))
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:top_n] if top_n else predictions
