import unittest
import os
import sys
import tempfile
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.content_based import ContentBasedModel

class TestContentBasedModel(unittest.TestCase):
    """
    Тесты для контентной модели рекомендаций.
    
    Проверяет корректность работы контентной модели, которая генерирует
    рекомендации на основе сходства контента фильмов с предпочтениями пользователя.
    """
    
    def setUp(self):
        """
        Подготовка тестовых данных перед каждым тестом.
        Создает тестовые данные о фильмах и рейтингах.
        """
        # Создание временной директории для хранения модели
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, 'test_model.pkl')
        
        # Создание тестовых данных о фильмах
        self.movies_data = pd.DataFrame({
            'movieId': [1, 2, 3, 4, 5],  # Добавили больше фильмов
            'plot': ['Action movie plot', 'Comedy movie plot', 'Drama movie plot', 'Sci-Fi movie plot', 'Horror movie plot'],
            'genres': ['Action,Adventure', 'Comedy,Romance', 'Drama,Thriller', 'Sci-Fi,Action', 'Horror,Thriller'],
            'directors': ['Director1', 'Director2', 'Director3', 'Director4', 'Director5'],
            'writers': ['Writer1', 'Writer2', 'Writer3', 'Writer4', 'Writer5'],
            'actors': ['Actor1,Actor2', 'Actor3,Actor4', 'Actor5,Actor6', 'Actor7,Actor8', 'Actor9,Actor10'],
            'countries': ['USA', 'UK', 'France', 'Germany', 'Japan'],
            'start_year': [2000, 2010, 2020, 2015, 2018],
            'type': ['movie', 'movie', 'movie', 'movie', 'movie']
        })
        
        # Создание тестовых данных рейтингов
        self.ratings_data = pd.DataFrame({
            'userId': [1, 1, 2, 2, 1],  # Добавили больше рейтингов
            'movieId': [1, 2, 1, 3, 4],
            'rating': [5.0, 4.0, 3.0, 5.0, 4.5],
            'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']
        })
        
        # Инициализация модели
        self.model = ContentBasedModel(model_path=self.model_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_model_initialization(self):
        """
        Проверяет корректность инициализации модели.
        """
        self.assertIsNone(self.model.feature_matrix)
        self.assertIsNone(self.model.movie_ids)
        self.assertEqual(self.model.model_path, self.model_path)
        self.assertEqual(self.model.user_profiles, {})
        self.assertEqual(self.model.watched_movies, {})

    def test_model_fit(self):
        """
        Проверяет корректность обучения модели.
        """
        self.model.fit(self.movies_data, self.ratings_data)
        
        # Проверка обучения модели
        self.assertIsNotNone(self.model.feature_matrix)
        self.assertIsNotNone(self.model.movie_ids)
        self.assertTrue(os.path.exists(self.model_path))
        
        # Проверка создания профилей пользователей
        self.assertIn(1, self.model.user_profiles)
        self.assertIn(2, self.model.user_profiles)
        
        # Проверка записи просмотренных фильмов
        self.assertIn(1, self.model.watched_movies)
        self.assertIn(2, self.model.watched_movies)

    def test_fit_with_empty_data(self):
        """Тест обучения модели с пустыми данными"""
        empty_movies = pd.DataFrame(columns=self.movies_data.columns)
        empty_ratings = pd.DataFrame(columns=self.ratings_data.columns)

        with self.assertRaises(ValueError):
            self.model.fit(empty_movies, self.ratings_data)

        with self.assertRaises(ValueError):
            self.model.fit(self.movies_data, empty_ratings)

    def test_model_save_load(self):
        """Тест сохранения и загрузки модели"""
        # Обучение и сохранение модели
        self.model.fit(self.movies_data, self.ratings_data)
        
        # Создание нового экземпляра и загрузка
        new_model = ContentBasedModel(model_path=self.model_path)
        new_model.load_model()
        
        # Проверка загруженной модели
        self.assertIsNotNone(new_model.feature_matrix)
        self.assertIsNotNone(new_model.movie_ids)
        self.assertEqual(len(new_model.user_profiles), 2)
        self.assertEqual(len(new_model.watched_movies), 2)

    def test_predict(self):
        """
        Проверяет корректность генерации рекомендаций.
        """
        self.model.fit(self.movies_data, self.ratings_data)
        
        # Тест предсказаний для существующего пользователя
        predictions = self.model.predict(user_id=1, top_n=2)
        self.assertEqual(len(predictions), 2)
        self.assertTrue(all(isinstance(pred[0], int) for pred in predictions))
        self.assertTrue(all(isinstance(pred[1], float) for pred in predictions))
        
        # Тест предсказаний для несуществующего пользователя
        with self.assertRaises(ValueError):
            self.model.predict(user_id=999)

    def test_get_similar_movies(self):
        """Тест поиска похожих фильмов"""
        self.model.fit(self.movies_data, self.ratings_data)
        
        # Тест поиска похожих фильмов
        similar = self.model.get_similar_movies(movie_id=1, top_n=2)
        self.assertEqual(len(similar), 2)
        self.assertTrue(all(isinstance(sim[0], int) for sim in similar))
        self.assertTrue(all(isinstance(sim[1], float) for sim in similar))
        
        # Тест с несуществующим фильмом
        with self.assertRaises(ValueError):
            self.model.get_similar_movies(movie_id=999)

    def test_recommend_recent(self):
        """Тест рекомендаций на основе недавно просмотренных фильмов"""
        self.model.fit(self.movies_data, self.ratings_data)
        
        # Тест рекомендаций на основе недавних просмотров
        recent = self.model.recommend_recent(user_id=1, top_n=2, recent_k=2)
        self.assertEqual(len(recent), 2)
        self.assertTrue(all(isinstance(rec[0], int) for rec in recent))
        self.assertTrue(all(isinstance(rec[1], float) for rec in recent))
        
        # Тест с несуществующим пользователем
        with self.assertRaises(ValueError):
            self.model.recommend_recent(user_id=999)

    def test_recommend_recent_with_few_recent_movies(self):
        """Тест рекомендаций, когда у пользователя меньше просмотренных фильмов, чем recent_k"""
        self.model.fit(self.movies_data, self.ratings_data)

        # Пользователь 2 посмотрел только 2 фильма
        recs = self.model.recommend_recent(user_id=2, top_n=2, recent_k=5)
        self.assertTrue(len(recs) <= 2)

    def test_predict_all_movies_watched(self):
        """Тест предсказаний, когда все фильмы уже просмотрены пользователем"""
        self.model.fit(self.movies_data, self.ratings_data)

        # Принудительно делаем вид, что пользователь 1 смотрел все фильмы
        self.model.watched_movies[1] = set(self.model.movie_ids)

        predictions = self.model.predict(user_id=1, top_n=3)
        self.assertEqual(predictions, [])

    def test_add_movie(self):
        """Тест добавления нового фильма в модель"""
        self.model.fit(self.movies_data, self.ratings_data)
        
        # Тест добавления нового фильма
        new_movie = {
            'movieId': 6,
            'plot': 'New movie plot',
            'genres': 'Action,Comedy',
            'directors': 'Director6',
            'writers': 'Writer6',
            'actors': 'Actor11,Actor12',
            'countries': 'USA',
            'start_year': 2023,
            'type': 'movie'
        }
        
        self.model.add_movie(new_movie)
        self.assertEqual(len(self.model.movie_ids), 6)
        self.assertEqual(self.model.feature_matrix.shape[0], 6)

    def test_add_existing_movie(self):
        """Тест добавления фильма с существующим movieId"""
        self.model.fit(self.movies_data, self.ratings_data)

        existing_movie = self.movies_data.iloc[0].to_dict()
        with self.assertRaises(ValueError):
            self.model.add_movie(existing_movie)

    def test_remove_movie(self):
        """Тест удаления фильма из модели"""
        self.model.fit(self.movies_data, self.ratings_data)
        
        # Тест удаления фильма
        self.model.remove_movie(movie_id=1)
        self.assertEqual(len(self.model.movie_ids), 4)
        self.assertEqual(self.model.feature_matrix.shape[0], 4)
        
        # Тест удаления несуществующего фильма
        with self.assertRaises(ValueError):
            self.model.remove_movie(movie_id=999)

    def test_update_movie(self):
        """Тест обновления информации о фильме"""
        self.model.fit(self.movies_data, self.ratings_data)
        
        # Тест обновления фильма
        updated_movie = {
            'movieId': 1,
            'plot': 'Updated plot',
            'genres': 'Action,Thriller',
            'directors': 'Director1',
            'writers': 'Writer1',
            'actors': 'Actor1,Actor2',
            'countries': 'USA',
            'start_year': 2001,
            'type': 'movie'
        }
        
        self.model.update_movie(updated_movie)
        self.assertEqual(len(self.model.movie_ids), 5)
        self.assertEqual(self.model.feature_matrix.shape[0], 5)

if __name__ == '__main__':
    unittest.main() 