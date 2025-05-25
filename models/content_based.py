import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.metrics.pairwise import cosine_similarity
from config import CONTENT_MODEL_PATH, CONTENT_TEST_MODEL_PATH
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

        if self.model_exists():
            self.load_model()
        else:
            print(f"[INFO] Контентная модель не найдена по пути {self.model_path}. Нужно вызвать fit().")

    def fit(self, movies_df, ratings_df, like_threshold=5.0, min_threshold=3.0):
        """
        Строит матрицу признаков фильмов и пользовательские профили.
        
        Профиль пользователя - это взвешенная сумма векторов признаков всех просмотренных фильмов.
        Вес каждого фильма зависит от рейтинга:
        - Фильмы с рейтингом >= like_threshold (5.0) имеют максимальный вес 1.0
        - Фильмы с рейтингом <= min_threshold (3.0) имеют нулевой вес
        - Фильмы с рейтингом между min_threshold и like_threshold имеют вес пропорциональный рейтингу
        
        Args:
            movies_df: DataFrame с информацией о фильмах
            ratings_df: DataFrame с рейтингами пользователей
            like_threshold: Максимальный рейтинг (по умолчанию 5.0)
            min_threshold: Минимальный рейтинг для учета фильма (по умолчанию 3.0)
        """
        if movies_df.empty:
            raise ValueError("Список с фильмами пуст")
        if ratings_df.empty:
            raise ValueError("Список с рейтингами пуст")

        self.preprocessor = build_preprocessor()
        # Преобразуем movieId в целое число
        movies_df = movies_df.copy()
        movies_df['movieId'] = movies_df['movieId'].astype(int)
        self.movie_ids = movies_df['movieId'].values
        self.feature_matrix = self.preprocessor.fit_transform(movies_df)
        ratings_df = ratings_df.sort_values(['userId', 'date'])

        # Создаем словарь для быстрого поиска индексов фильмов
        movie_id_to_idx = {int(mid): idx for idx, mid in enumerate(self.movie_ids)}
        
        # Группируем рейтинги по пользователям
        for user_id, group in ratings_df.groupby('userId'):
            # Сохраняем список просмотренных фильмов
            watched_ids = [int(mid) for mid in group['movieId'].tolist()]
            self.watched_movies[user_id] = watched_ids
            
            # Собираем индексы фильмов и их рейтинги
            movie_indices = []
            ratings = []
            for mid, rating in zip(group['movieId'], group['rating']):
                if mid in movie_id_to_idx:
                    movie_indices.append(movie_id_to_idx[mid])
                    
                    # Преобразуем рейтинг в вес от 0 до 1
                    # Рейтинг 5.0 -> вес 1.0
                    # Рейтинг 3.0 -> вес 0.0
                    # Рейтинг 4.0 -> вес 0.5
                    normalized_rating = (rating - min_threshold) / (like_threshold - min_threshold)
                    normalized_rating = max(0, min(1, normalized_rating))  # ограничиваем диапазон [0, 1]
                    ratings.append(normalized_rating)
            
            if movie_indices:
                # Получаем векторы признаков для просмотренных фильмов
                movie_vectors = self.feature_matrix[movie_indices]
                
                # Создаем взвешенный профиль пользователя
                # 1. Преобразуем веса в столбец для умножения
                weights = np.array(ratings).reshape(-1, 1)
                
                # 2. Умножаем каждый вектор признаков фильма на соответствующий вес
                # Фильмы с высоким рейтингом будут иметь больший вклад в профиль
                weighted_vectors = movie_vectors.multiply(weights)
                
                # 3. Суммируем все взвешенные векторы
                # Получаем один вектор, представляющий предпочтения пользователя
                profile = weighted_vectors.sum(axis=0)
                
                # 4. Преобразуем к csr_matrix для совместимости
                profile = csr_matrix(profile)
                
                # 5. Нормализуем профиль, деля на количество фильмов
                # Это нужно, чтобы профили пользователей с разным количеством оценок
                # были сравнимы между собой
                profile = profile.multiply(1.0 / len(movie_indices))
                
                self.user_profiles[user_id] = profile

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
        
        # Нормализуем косинусное сходство из [-1, 1] в [0, 1]
        sims = (sims + 1) / 2

        known_movie_ids = set(self.watched_movies.get(user_id, []))
        candidates = [(int(mid), sim) for mid, sim in zip(self.movie_ids, sims) if mid not in known_movie_ids]
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
            (int(mid), sim) for mid, sim in zip(self.movie_ids, sims) if mid != movie_id
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
        candidates = [(int(mid), sim) for mid, sim in zip(self.movie_ids, sims) if mid not in already_seen]
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[:top_n]
    
    def add_movie(self, movie_dict: dict):
        """
        Добавляет новый фильм к матрице признаков без переобучения.
        movie_dict должен содержать все необходимые поля, как в movies_df.
        """
        if self.preprocessor is None:
            raise ValueError("Препроцессор не загружен. Нужно вызвать fit() или load_model().")

        if self.feature_matrix is None or self.movie_ids is None:
            raise ValueError("Модель не обучена. Вызови fit().")

        # Преобразуем словарь в DataFrame с одной строкой
        movie_row = pd.DataFrame([movie_dict])
        movie_id = int(movie_row['movieId'].values[0])

        # Проверяем, не существует ли уже фильм с таким ID
        if movie_id in self.movie_ids:
            raise ValueError(f"Фильм с ID {movie_id} уже существует в модели")

        # Только реально используемые признаки:
        required_columns = [tr[2] for tr in self.preprocessor.transformers]
        missing_cols = set(required_columns) - set(movie_row.columns)
        if missing_cols:
            raise ValueError(f"Отсутствуют обязательные столбцы: {missing_cols}")

        if 'start_year' in movie_row.columns:
            movie_row['start_year'] = movie_row['start_year'].astype(float)
            
        # Трансформация признаков нового фильма
        new_vector = self.preprocessor.transform(movie_row)

        # Обновим матрицу признаков и movie_ids
        self.feature_matrix = vstack([self.feature_matrix, new_vector])
        self.movie_ids = np.append(self.movie_ids, movie_id)

        print(f"[INFO] Добавлен фильм ID {movie_id}. Обновлена матрица признаков.")

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

    def update_movie(self, movie_dict: dict):
        """
        Обновляет признаки фильма по movie_id.
        movie_dict — словарь с ключами, как в movies_df.
        """
        if self.movie_ids is None or self.feature_matrix is None:
            raise RuntimeError("Модель не загружена или не обучена.")

        movie_df = pd.DataFrame([movie_dict])
        movie_id = int(movie_dict.get('movieId'))

        if movie_id is None:
            raise ValueError("[ERROR] В словаре отсутствует ключ 'movieId'.")

        if movie_id not in self.movie_ids:
            raise ValueError(f"[ERROR] Фильм {movie_id} не найден в модели.")
        
        if 'start_year' in movie_df.columns:
            movie_df['start_year'] = movie_df['start_year'].astype(float)

        idx = np.where(self.movie_ids == movie_id)[0][0]

        # Преобразуем обновлённые признаки в вектор
        new_features = self.preprocessor.transform(movie_df)

        # Обновим строку в матрице признаков
        self.feature_matrix[idx] = new_features

        print(f"[INFO] Обновлены признаки фильма ID {movie_id}.")

    
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