import unittest
import os
import os
import sys
import tempfile
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.collaborative import CollaborativeModel

class TestCollaborativeModel(unittest.TestCase):
    """
    Тесты для коллаборативной модели рекомендаций.
    
    Проверяет корректность работы коллаборативной модели, которая генерирует
    рекомендации на основе сходства пользователей и их оценок фильмов.
    """
    
    def setUp(self):
        """
        Подготовка тестовых данных перед каждым тестом.
        Создает тестовые данные о рейтингах фильмов.
        """
        # Создание временной директории для хранения модели
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, 'test_model.pkl')
        
        # Создание тестовых данных рейтингов
        self.ratings_data = pd.DataFrame({
            'userId': [1, 1, 2, 2, 1],
            'movieId': [1, 2, 1, 3, 4],
            'rating': [5.0, 4.0, 3.0, 5.0, 4.5],
            'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']
        })
        
        # Инициализация модели
        self.model = CollaborativeModel(model_path=self.model_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_model_initialization(self):
        """
        Проверяет корректность инициализации модели.
        """
        self.assertIsNone(self.model.model)
        self.assertEqual(self.model.model_path, self.model_path)
        self.assertEqual(self.model.all_movie_ids, [])
        self.assertEqual(self.model.user_rated_movies, {})

    def test_model_fit(self):
        """
        Проверяет корректность обучения модели.
        """
        self.model.fit(self.ratings_data)
        
        # Проверка обучения модели
        self.assertIsNotNone(self.model.model)
        self.assertTrue(os.path.exists(self.model_path))
        
        # Проверка сбора ID фильмов
        self.assertEqual(len(self.model.all_movie_ids), 3)
        self.assertTrue(all(mid in self.model.all_movie_ids for mid in [1, 2, 3]))
        
        # Проверка сбора рейтингов пользователей
        self.assertEqual(len(self.model.user_rated_movies[1]), 2)
        self.assertEqual(len(self.model.user_rated_movies[2]), 2)
        self.assertEqual(len(self.model.user_rated_movies[3]), 2)

    def test_model_save_load(self):
        """
        Проверяет корректность сохранения и загрузки модели.
        """
        # Обучение и сохранение модели
        self.model.fit(self.ratings_data)
        
        # Создание нового экземпляра и загрузка
        new_model = CollaborativeModel(model_path=self.model_path)
        new_model.load_model()
        
        # Проверка загруженной модели
        self.assertIsNotNone(new_model.model)
        self.assertEqual(len(new_model.all_movie_ids), 3)
        self.assertEqual(len(new_model.user_rated_movies), 3)

    def test_load_model_called_on_init_if_exists(self):
        """Тест, что модель загружается при инициализации, если файл существует"""
        self.model.fit(self.ratings_data)

        # создаём новую модель — она должна загрузиться автоматически
        new_model = CollaborativeModel(model_path=self.model_path)
        self.assertIsNotNone(new_model.model)

    def test_predict(self):
        """
        Проверяет корректность генерации рекомендаций.
        """
        self.model.fit(self.ratings_data)
        
        # Тест предсказаний для существующего пользователя
        predictions = self.model.predict(user_id=1, top_n=2)
        self.assertGreater(len(predictions), 0)  # Проверяем, что есть хотя бы одно предсказание
        self.assertTrue(all(isinstance(pred[0], int) for pred in predictions))
        self.assertTrue(all(isinstance(pred[1], float) for pred in predictions))
        
        # Тест предсказаний для несуществующего пользователя
        with self.assertRaises(ValueError):
            self.model.predict(user_id=999)

    def test_predict_excludes_seen_movies(self):
        """Тест, что предсказания не содержат фильмов, уже просмотренных пользователем"""
        self.model.fit(self.ratings_data)
        seen = self.model.user_rated_movies[1]
        predictions = self.model.predict(user_id=1)
        predicted_ids = [mid for mid, _ in predictions]

        for movie_id in predicted_ids:
            self.assertNotIn(movie_id, seen)

    def test_predict_without_training(self):
        """Тест попытки предсказания без предварительного обучения модели"""
        with self.assertRaises(RuntimeError):
            self.model.predict(user_id=1)

    def test_predict_user_with_no_ratings(self):
        """Тест, что предсказания не ломаются, если пользователь есть, но ничего не оценил"""
        new_data = self.ratings_data.copy()
        new_data = pd.concat([new_data, pd.DataFrame({'userId': [4], 'movieId': [99], 'rating': [np.nan]})], ignore_index=True)
        new_data = new_data.dropna()  # пользователь 4 останется без записей
        self.model.fit(new_data)

        with self.assertRaises(ValueError):
            self.model.predict(user_id=4)

    def test_predict_user_with_all_movies_seen(self):
        """Тест, что предсказания пусты, если пользователь видел все фильмы"""
        self.model.fit(self.ratings_data)
        # пользователь 1 видел фильмы 1 и 2 → добавим недостающий
        extended_data = pd.concat([self.ratings_data, pd.DataFrame({'userId': [1], 'movieId': [3], 'rating': [5.0]})], ignore_index=True)
        self.model.fit(extended_data)

        predictions = self.model.predict(user_id=1)
        self.assertEqual(predictions, [])

if __name__ == '__main__':
    unittest.main() 