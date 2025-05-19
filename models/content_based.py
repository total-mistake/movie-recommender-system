import os
import pickle
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from config import CONTENT_MODEL_PATH
from .base import BaseModel
from .preprocessing import build_preprocessor
from scipy.sparse import csr_matrix

class ContentBasedModel(BaseModel):
    def __init__(self, model_path=CONTENT_MODEL_PATH):
        self.model_path = model_path
        self.movie_ids = None
        self.feature_matrix = None
        self.user_profiles = {}
        self.preprocessor = None
        self.watched_movies = {}

        if os.path.exists(self.model_path):
            self.load_model()
        else:
            print(f"[INFO] Контентная модель не найдена по пути {self.model_path}. Нужно вызвать fit().")

    def fit(self, movies_df, ratings_df, like_threshold=5.0):
        """
        Строит матрицу признаков фильмов и пользовательские профили
        только по понравившимся фильмам (рейтинг >= like_threshold).
        """
        self.preprocessor = build_preprocessor()
        self.movie_ids = movies_df['movieId'].values
        self.feature_matrix = self.preprocessor.fit_transform(movies_df)
        ratings_df = ratings_df.sort_values(['userId', 'date'])

        for user_id, group in ratings_df.groupby('userId'):
            print(f"[INFO] Обработка пользователя {user_id}...")
            watched_ids = group['movieId'].tolist()  # сохраняет порядок
            self.watched_movies[user_id] = watched_ids


            # Оставляем только те фильмы, которые пользователь оценил >= threshold
            liked = group[group['rating'] >= like_threshold]
            liked_ids = liked['movieId'].values

            # Индексы понравившихся фильмов
            watched_idxs = [np.where(self.movie_ids == mid)[0][0] for mid in liked_ids if mid in self.movie_ids]

            if watched_idxs:
                subset = self.feature_matrix[watched_idxs]
                # Суммируем строки (получим csr_matrix)
                sum_vector = subset.sum(axis=0)  # результат — это numpy.matrix

                # Преобразуем к csr_matrix (иначе .multiply не сработает)
                sum_vector = csr_matrix(sum_vector)

                # Делим на количество фильмов (поэлементное деление)
                profile_sparse = sum_vector.multiply(1.0 / subset.shape[0])

                self.user_profiles[user_id] = profile_sparse

        self._save_model()


    def _save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({
                'movie_ids': self.movie_ids,
                'feature_matrix': self.feature_matrix,
                'user_profiles': self.user_profiles,
                'watched_movies': self.watched_movies,
                'preprocessor': self.preprocessor
            }, f)

    def load_model(self):
        with open(self.model_path, "rb") as f:
            data = pickle.load(f)
            self.movie_ids = data['movie_ids']
            self.feature_matrix = data['feature_matrix']
            self.user_profiles = data['user_profiles']
            self.watched_movies = data.get('watched_movies', {})
            self.preprocessor = data['preprocessor']
    
    def predict_(self, user_id):
        """
        Предсказывает оценки схожести для всех непросмотренных фильмов пользователя.
        Возвращает список (movie_id, score).
        """
        if self.feature_matrix is None or self.movie_ids is None:
            raise RuntimeError("Модель не обучена или не загружена.")

        if user_id not in self.user_profiles:
            raise ValueError(f"[ERROR] Пользователь {user_id} не найден в профилях.")

        # Индексы просмотренных фильмов
        known_movie_ids = set(
            mid for mid_idx, mid in enumerate(self.movie_ids)
            if cosine_similarity(self.user_profiles[user_id], self.feature_matrix[mid_idx])[0][0] > 0.9999
        )

        # Кандидаты — непросмотренные фильмы
        candidates = [(idx, mid) for idx, mid in enumerate(self.movie_ids) if mid not in known_movie_ids]

        user_vector = self.user_profiles[user_id]
        predictions = [
            (mid, cosine_similarity(user_vector, self.feature_matrix[idx])[0][0])
            for idx, mid in candidates
        ]
        return predictions
    
    def predict(self, user_id, top_n=None):
        if self.feature_matrix is None or self.movie_ids is None:
            raise RuntimeError("Модель не обучена или не загружена.")

        if user_id not in self.user_profiles:
            raise ValueError(f"[ERROR] Пользователь {user_id} не найден в профилях.")

        user_vector = self.user_profiles[user_id]
        sims = cosine_similarity(user_vector, self.feature_matrix)[0]

        known_movie_ids = set(self.watched_movies.get(user_id, []))
        candidates = [(mid, sim) for mid, sim in zip(self.movie_ids, sims) if mid not in known_movie_ids]
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[:top_n] if top_n else candidates
    
    # def predict(self, user_id, top_n=None):
    #     if self.feature_matrix is None or self.movie_ids is None:
    #         raise RuntimeError("Модель не обучена или не загружена.")

    #     if user_id not in self.user_profiles:
    #         raise ValueError(f"[ERROR] Пользователь {user_id} не найден в профилях.")

    #     user_vector = self.user_profiles[user_id]
    #     start = time.time()
    #     sims = cosine_similarity(user_vector, self.feature_matrix)[0]

    #     start = time.time()
    #     known_movie_ids = set(self.watched_movies.get(user_id, []))
    #     mask = np.isin(self.movie_ids, list(known_movie_ids), invert=True)
    #     filtered_ids = self.movie_ids[mask]
    #     filtered_sims = sims[mask]
    #     print(f"[DEBUG] Фильтрация заняла {time.time() - start:.2f} сек")
    #     start = time.time()
    #     candidates = np.column_stack((filtered_ids, filtered_sims))  # shape: (n_samples, 2)

    #     # Сортировка по similarity (второй колонке), по убыванию
    #     sorted_indices = np.argsort(-candidates[:, 1])
    #     candidates = candidates[sorted_indices]
    #     print(f"[DEBUG] Сортировка заняла {time.time() - start:.2f} сек")

    #     return candidates[:top_n] if top_n else candidates
    
    def get_similar_movies(self, movie_id, top_n=10):
        """
        Возвращает top_n наиболее похожих фильмов на переданный movie_id.
        """
        if self.feature_matrix is None or self.movie_ids is None:
            raise RuntimeError("Модель не обучена или не загружена.")

        if movie_id not in self.movie_ids:
            raise ValueError(f"[ERROR] Фильм {movie_id} не найден в модели.")

        idx = np.where(self.movie_ids == movie_id)[0][0]
        movie_vector = self.feature_matrix[idx]
        sims = cosine_similarity(movie_vector, self.feature_matrix)[0]

        similar = [
            (mid, sim) for mid, sim in zip(self.movie_ids, sims) if mid != movie_id
        ]
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar[:top_n]
    
    def recommend_recent(self, user_id, top_n=10, recent_k=3):
        """
        Рекомендует фильмы на основе recent_k последних просмотренных пользователем фильмов.
        Неважно, положительная или отрицательная была оценка.
        """
        if user_id not in self.watched_movies or not self.watched_movies[user_id]:
            raise ValueError(f"[ERROR] Нет данных о просмотренных фильмах пользователя {user_id}.")

        # Получим последние K просмотренных фильмов
        recent = self.watched_movies[user_id][-recent_k:]  # последние просмотренные, порядок важен

        # Найдём индексы этих фильмов
        idxs = [np.where(self.movie_ids == mid)[0][0] for mid in recent if mid in self.movie_ids]

        if not idxs:
            raise ValueError(f"[ERROR] Недавние фильмы не найдены в обучающей выборке.")

        # Строим временный профиль пользователя как среднее векторов признаков этих фильмов
        subset = self.feature_matrix[idxs]
        profile = subset.mean(axis=0)
        profile = csr_matrix(profile)  # убедимся, что это sparse для cosine_similarity

        sims = cosine_similarity(profile, self.feature_matrix)[0]

        # Исключим уже просмотренные
        already_seen = set(self.watched_movies.get(user_id, []))
        candidates = [(mid, sim) for mid, sim in zip(self.movie_ids, sims) if mid not in already_seen]
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[:top_n]




