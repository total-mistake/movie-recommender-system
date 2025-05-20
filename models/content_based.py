import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import vstack
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
    
    def add_movie(self, movie_row: pd.DataFrame):
        """
        Добавляет новый фильм (одна строка DataFrame) к матрице признаков без переобучения.
        movie_row должен иметь те же столбцы, что и movies_df из .fit()
        """
        if self.preprocessor is None:
            raise ValueError("Препроцессор не загружен. Нужно вызвать fit() или load_model().")

        if self.feature_matrix is None or self.movie_ids is None:
            raise ValueError("Модель не обучена. Вызови fit().")

        if len(movie_row) != 1:
            raise ValueError("Передай DataFrame с ровно одной строкой.")

        # Проверим наличие нужных колонок
        required_columns = self.preprocessor.feature_names_in_
        missing_cols = set(required_columns) - set(movie_row.columns)
        if missing_cols:
            raise ValueError(f"Отсутствуют обязательные столбцы: {missing_cols}")

        # Трансформация признаков нового фильма
        new_vector = self.preprocessor.transform(movie_row)

        # Обновим матрицу признаков и movie_ids
        self.feature_matrix = vstack([self.feature_matrix, new_vector])
        self.movie_ids = np.append(self.movie_ids, movie_row['movieId'].values[0])

        print(f"[INFO] Добавлен фильм ID {movie_row['movieId'].values[0]}. Обновлена матрица признаков.")

    def remove_movie(self, movie_id):
        """
        Удаляет фильм из модели по его movie_id.
        """
        if self.movie_ids is None or self.feature_matrix is None:
            raise RuntimeError("Модель не загружена или не обучена.")
        
        if movie_id not in self.movie_ids:
            raise ValueError(f"[ERROR] Фильм {movie_id} не найден в модели.")

        idx = np.where(self.movie_ids == movie_id)[0][0]

        # Удаляем строку из feature_matrix и id
        self.feature_matrix = vstack([
            self.feature_matrix[:idx],
            self.feature_matrix[idx+1:]
        ])
        self.movie_ids = np.delete(self.movie_ids, idx)

        # Удалим этот фильм из просмотренных у всех пользователей
        for uid in self.watched_movies:
            self.watched_movies[uid] = [mid for mid in self.watched_movies[uid] if mid != movie_id]

    def update_movie(self, movie_df):
        """
        Обновляет признаки фильма по movie_id.
        movie_df — DataFrame с одной строкой (аналогичный строке из movies_df).
        """
        if self.movie_ids is None or self.feature_matrix is None:
            raise RuntimeError("Модель не загружена или не обучена.")

        if len(movie_df) != 1:
            raise ValueError("[ERROR] Ожидается DataFrame с одной строкой.")

        movie_id = movie_df.iloc[0]['movieId']
        if movie_id not in self.movie_ids:
            raise ValueError(f"[ERROR] Фильм {movie_id} не найден в модели.")

        idx = np.where(self.movie_ids == movie_id)[0][0]

        # Преобразуем фильм в вектор
        new_features = self.preprocessor.transform(movie_df)

        # Обновим строку в матрице признаков
        self.feature_matrix[idx] = new_features

    def update_user_profile(self, user_id, new_movie_id, new_rating):
        """
        Обновляет профиль пользователя при появлении новой оценки.

        new_movie_id — идентификатор фильма
        new_rating — оценка пользователя (например, от 1 до 5)
        """
        if self.user_profiles is None or self.feature_matrix is None:
            raise RuntimeError("Модель не обучена или не загружена.")

        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = np.zeros(self.feature_matrix.shape[1])

        # Найдём индекс фильма
        try:
            idx = np.where(self.movie_ids == new_movie_id)[0][0]
        except IndexError:
            raise ValueError(f"Фильм {new_movie_id} не найден в модели.")

        movie_vec = self.feature_matrix[idx]

        # Обновим профиль пользователя — можно взять простое средневзвешенное обновление
        # (например, сглаживание предыдущего профиля с новым фильмом, взвешенное по рейтингу)
        old_profile = self.user_profiles[user_id]
        alpha = 0.1  # скорость адаптации (можно настраивать)

        new_profile = (1 - alpha) * old_profile + alpha * new_rating * movie_vec
        self.user_profiles[user_id] = new_profile

    def add_new_user(self, user_id, favorite_genres: list[str]):
        """
        Создает профиль пользователя на основе любимых жанров.
        Все остальные признаки задаются пустыми/нулевыми.
        """
        dummy_movie = {
            'plot': '',
            'genres': ','.join(favorite_genres),
            'directors': '',
            'writers': '',
            'actors': '',
            'countries': '',
            'start_year': 0,
            'type': ''
        }
        dummy_df = pd.DataFrame([dummy_movie])
        user_vector = self.preprocessor.transform(dummy_df)
        self.user_profiles[user_id] = user_vector[0]