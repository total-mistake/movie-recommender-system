import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import vstack, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from config import CONTENT_MODEL_PATH
from .base import BaseModel
from .preprocessing import build_preprocessor
import time
from datetime import timedelta

class ContentBasedModel(BaseModel):
    def __init__(self, model_path=CONTENT_MODEL_PATH, l2_reg=0.1):
        self.model_path = model_path
        self.l2_reg = l2_reg  # Коэффициент L2-регуляризации
        self.movie_ids = None
        self.feature_matrix = None
        self.user_profiles = {}
        self.preprocessor = None
        self.watched_movies = {}
        self.movie_ratings = {}

        if self.model_exists():
            self.load_model()
        else:
            print(f"[INFO] Контентная модель не найдена по пути {self.model_path}. Нужно вызвать fit().")

    def _l2_regularization(self, vector):
        """
        Применяет L2-регуляризацию к вектору.
        Регуляризация уменьшает влияние больших значений и делает профиль более устойчивым.
        """
        if isinstance(vector, csr_matrix):
            # Для разреженных матриц
            squared_sum = vector.multiply(vector).sum()
            if squared_sum > 0:
                scale = 1.0 / (1.0 + self.l2_reg * squared_sum)
                return vector.multiply(scale)
            return vector
        else:
            # Для плотных матриц
            squared_sum = np.sum(vector ** 2)
            if squared_sum > 0:
                scale = 1.0 / (1.0 + self.l2_reg * squared_sum)
                return vector * scale
            return vector

    def fit(self, movies_df, ratings_df, like_threshold=5.0, min_threshold=3.0):
        """
        Строит матрицу признаков фильмов и пользовательские профили.
        Сохраняет информацию о рейтингах фильмов для построения профилей новых пользователей.
        """
        if movies_df.empty:
            raise ValueError("Список с фильмами пуст")
        if ratings_df.empty:
            raise ValueError("Список с рейтингами пуст")

        # Сохраняем существующие профили пользователей без рейтингов
        existing_profiles = {
            uid: profile for uid, profile in self.user_profiles.items() 
            if uid not in ratings_df['User_ID'].values
        }

        # Инициализируем препроцессор и строим матрицу признаков
        self.preprocessor = build_preprocessor()
        movies_df = movies_df.copy()
        movies_df['Movie_ID'] = movies_df['Movie_ID'].astype(int)
        self.movie_ids = movies_df['Movie_ID'].values
        self.feature_matrix = self.preprocessor.fit_transform(movies_df)
        
        # Сохраняем информацию о рейтингах фильмов
        movie_ratings = ratings_df.groupby('Movie_ID').agg({
            'Rating': ['mean', 'count']
        }).reset_index()
        self.movie_ratings = {
            int(row['Movie_ID'].iloc[0]): (row[('Rating', 'mean')], row[('Rating', 'count')])
            for _, row in movie_ratings.iterrows()
        }
        
        # Строим профили пользователей с рейтингами
        ratings_df = ratings_df.sort_values(['User_ID', 'Date_Rated'])
        movie_id_to_idx = {int(mid): idx for idx, mid in enumerate(self.movie_ids)}
        
        # Очищаем watched_movies и заполняем заново из рейтингов
        self.watched_movies = {}
        
        for user_id, group in ratings_df.groupby('User_ID'):
            watched_ids = [int(mid) for mid in group['Movie_ID'].tolist()]
            self.watched_movies[user_id] = watched_ids
            
            movie_indices = []
            ratings = []
            
            for mid, rating in zip(group['Movie_ID'], group['Rating']):
                if mid in movie_id_to_idx:
                    movie_indices.append(movie_id_to_idx[mid])
                    normalized_rating = (rating - min_threshold) / (like_threshold - min_threshold)
                    normalized_rating = max(0, min(1, normalized_rating))
                    ratings.append(normalized_rating)
            
            if movie_indices:
                movie_vectors = self.feature_matrix[movie_indices]
                weights = np.array(ratings).reshape(-1, 1)
                weighted_vectors = movie_vectors.multiply(weights)
                profile = weighted_vectors.sum(axis=0)
                profile = csr_matrix(profile)
                profile = self._l2_regularization(profile)
                self.user_profiles[user_id] = profile

        # Восстанавливаем профили пользователей без рейтингов
        self.user_profiles.update(existing_profiles)

        self._save_model()

    def _calculate_popularity_score(self, avg_rating, rating_count, min_ratings=1000):
        """
        Вычисляет единый скор популярности фильма на основе рейтинга и количества оценок.
        Использует формулу Байеса для учета количества оценок.
        
        Args:
            avg_rating: средний рейтинг фильма
            rating_count: количество оценок
            min_ratings: минимальное количество оценок для учета
        
        Returns:
            float: скор популярности от 0 до 1
        """
        if rating_count < min_ratings:
            return 0.0
            
        # Нормализуем рейтинг от 1 до 5 в диапазон 0-1
        normalized_rating = (avg_rating - 1) / 4
        
        # Вычисляем доверительный интервал для рейтинга
        # Чем больше оценок, тем ближе к реальному рейтингу
        confidence = min(1.0, rating_count / 10000)  # максимум при 1000 оценках
        
        # Комбинируем рейтинг и количество оценок
        popularity = normalized_rating * confidence
        
        return popularity

    def add_new_user(self, user_id, favorite_genres: list[str], min_rating=3.5, min_reviews=1000):
        """
        Создает профиль пользователя на основе топ фильмов в любимых жанрах.
        
        Args:
            user_id: ID пользователя
            favorite_genres: список любимых жанров
            min_rating: минимальный средний рейтинг фильма (от 1 до 5)
            min_reviews: минимальное количество отзывов
        """
        start_time = time.time()
        
        if not self.movie_ratings:
            raise ValueError("Нет информации о рейтингах фильмов. Нужно вызвать fit() с рейтингами.")
        
        # Этап 1: Подготовка жанрового вектора пользователя
        stage1_start = time.time()
        genres_pipeline = self.preprocessor.transformers_[1][1]
        genres_binarizer = genres_pipeline.named_steps['binarizer'].mlb
        genres_str = ', '.join(favorite_genres)
        genre_vector = genres_pipeline.transform([genres_str])
        print(f"[Этап 1] Подготовка жанрового вектора: {timedelta(seconds=time.time()-stage1_start)}")
        
        # Этап 2: Предварительная фильтрация и поиск валидных фильмов
        stage2_start = time.time()
        
        # Создаем массив рейтингов и количества оценок для всех фильмов
        ratings_array = np.zeros((len(self.movie_ids), 2))
        for i, movie_id in enumerate(self.movie_ids):
            if movie_id in self.movie_ratings:
                ratings_array[i] = self.movie_ratings[movie_id]
        
        # Применяем пороговые значения для быстрой фильтрации
        avg_ratings = ratings_array[:, 0]
        rating_counts = ratings_array[:, 1]
        
        # Создаем маску для фильмов, прошедших базовую фильтрацию
        base_mask = (avg_ratings >= min_rating) & (rating_counts >= min_reviews)
        
        if not np.any(base_mask):
            raise ValueError(f"Не найдено фильмов с рейтингом >= {min_rating} и количеством отзывов >= {min_reviews}")
        
        # Получаем индексы фильмов, прошедших базовую фильтрацию
        base_indices = np.where(base_mask)[0]
        
        # Вычисляем популярность только для отфильтрованных фильмов
        filtered_ratings = avg_ratings[base_indices]
        filtered_counts = rating_counts[base_indices]
        
        # Вычисляем популярность для отфильтрованных фильмов
        normalized_ratings = (filtered_ratings - 1) / 4
        confidence = np.minimum(1.0, filtered_counts / 10000)
        popularity_scores = normalized_ratings * confidence
        
        # Получаем жанровую часть векторов для всех отфильтрованных фильмов сразу
        plot_dim = self.preprocessor.transformers_[0][1].named_steps['lsa'].n_components
        genres_start = plot_dim
        genres_end = genres_start + len(genres_binarizer.classes_)
        
        # Получаем все необходимые данные для отфильтрованных фильмов
        filtered_movie_genres = self.feature_matrix[base_indices, genres_start:genres_end]
        filtered_movie_vectors = self.feature_matrix[base_indices]
        filtered_movie_ids = self.movie_ids[base_indices]
        
        # Создаем список кортежей для дальнейшей обработки
        valid_movies = list(zip(filtered_movie_ids, popularity_scores, filtered_movie_genres, filtered_movie_vectors))
        
        print(f"[Этап 2] Поиск валидных фильмов: {timedelta(seconds=time.time()-stage2_start)}")
        print(f"Найдено валидных фильмов: {len(valid_movies)}")
        print(f"Отфильтровано фильмов: {len(self.movie_ids) - len(valid_movies)}")
        
        # Этап 3: Расчет схожести жанров (оптимизированная версия)
        stage3_start = time.time()
        
        # Вычисляем схожесть для всех фильмов сразу
        genre_similarities = cosine_similarity(genre_vector, filtered_movie_genres)[0]
        
        # Комбинируем схожесть с популярностью
        combined_scores = genre_similarities * popularity_scores
        
        # Создаем список кортежей (movie_id, score, vector) и сортируем
        scored_movies = list(zip(filtered_movie_ids, combined_scores, filtered_movie_vectors))
        scored_movies.sort(key=lambda x: x[1], reverse=True)
        top_movies = scored_movies[:10]
        
        print(f"[Этап 3] Расчет схожести и сортировка: {timedelta(seconds=time.time()-stage3_start)}")
        
        # Этап 4: Построение профиля пользователя
        stage4_start = time.time()
        profile = sum(movie_vector for _, _, movie_vector in top_movies) / len(top_movies)
        profile = csr_matrix(profile)
        profile = self._l2_regularization(profile)
        
        self.user_profiles[user_id] = profile
        self.watched_movies[user_id] = []
        print(f"[Этап 4] Построение профиля: {timedelta(seconds=time.time()-stage4_start)}")
        
        total_time = time.time() - start_time
        print(f"Общее время выполнения: {timedelta(seconds=total_time)}")

    def predict(self, user_id, top_n=None):
        if self.feature_matrix is None or self.movie_ids is None:
            raise RuntimeError("Модель не обучена или не загружена.")

        if user_id not in self.user_profiles:
            raise ValueError(f"[ERROR] Пользователь {user_id} не найден в профилях.")

        user_vector = self.user_profiles[user_id]
        sims = cosine_similarity(user_vector, self.feature_matrix)[0]

        known_movie_ids = set(self.watched_movies.get(user_id, []))
        candidates = [(int(mid), sim) for mid, sim in zip(self.movie_ids, sims) if mid not in known_movie_ids]
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[:top_n] if top_n else candidates

    def _save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({
                'movie_ids': self.movie_ids,
                'feature_matrix': self.feature_matrix,
                'user_profiles': self.user_profiles,
                'watched_movies': self.watched_movies,
                'preprocessor': self.preprocessor,
                'l2_reg': self.l2_reg,
                'movie_ratings': self.movie_ratings  # Добавляем сохранение рейтингов
            }, f)

    def load_model(self):
        with open(self.model_path, "rb") as f:
            data = pickle.load(f)
            self.movie_ids = data['movie_ids']
            self.feature_matrix = data['feature_matrix']
            self.user_profiles = data['user_profiles']
            self.watched_movies = data.get('watched_movies', {})
            self.preprocessor = data['preprocessor']
            self.l2_reg = data.get('l2_reg', 0.1)
            self.movie_ratings = data.get('movie_ratings', {})  # Добавляем загрузку рейтингов

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
        movie_id = int(movie_row['Movie_ID'].values[0])

        # Проверяем, не существует ли уже фильм с таким ID
        if movie_id in self.movie_ids:
            raise ValueError(f"Фильм с ID {movie_id} уже существует в модели")

        # Только реально используемые признаки:
        required_columns = [tr[2] for tr in self.preprocessor.transformers]
        missing_cols = set(required_columns) - set(movie_row.columns)
        if missing_cols:
            raise ValueError(f"Отсутствуют обязательные столбцы: {missing_cols}")

        if 'Year' in movie_row.columns:
            movie_row['Year'] = movie_row['Year'].astype(float)
            
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
        movie_id = int(movie_dict.get('Movie_ID'))

        if movie_id is None:
            raise ValueError("[ERROR] В словаре отсутствует ключ 'Movie_ID'.")

        if movie_id not in self.movie_ids:
            raise ValueError(f"[ERROR] Фильм {movie_id} не найден в модели.")
        
        if 'Year' in movie_df.columns:
            movie_df['Year'] = movie_df['Year'].astype(float)

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

        old_profile = self.user_profiles[user_id]
        alpha = 0.1  # скорость адаптации

        new_profile = (1 - alpha) * old_profile + alpha * new_rating * movie_vec
        self.user_profiles[user_id] = new_profile
