import unittest
import os
import sys
import tempfile
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.hybrid import HybridModel

class TestHybridModel(unittest.TestCase):
    """
    Тесты для гибридной модели рекомендаций.
    
    Проверяет корректность работы гибридной модели, которая комбинирует
    контентную и коллаборативную модели с заданным весом alpha.
    """
    
    def setUp(self):
        """
        Подготовка тестовых данных перед каждым тестом.
        Создает тестовые данные о фильмах и рейтингах.
        """
        # Создание тестовых данных о фильмах
        self.movies_data = pd.DataFrame({
            'movieId': [1, 2, 3, 4, 5],
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
            'userId': [1, 1, 2, 2, 1],
            'movieId': [1, 2, 1, 3, 4],
            'rating': [5.0, 4.0, 3.0, 5.0, 4.5],
            'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']
        })
        
        # Инициализация модели
        self.model = HybridModel(alpha=0.5)

    def test_model_initialization(self):
        """Тест корректной инициализации гибридной модели"""
        self.assertEqual(self.model.alpha, 0.5)
        self.assertIsNotNone(self.model.content_model)
        self.assertIsNotNone(self.model.collaborative_model)

    def test_model_fit(self):
        """Тест обучения гибридной модели и её компонентов"""
        self.model.fit(self.movies_data, self.ratings_data)
        
        # Проверка обучения внутренних моделей
        self.assertIsNotNone(self.model.content_model)
        self.assertIsNotNone(self.model.collaborative_model)

    def test_predict(self):
        """Тест предсказаний гибридной модели"""
        self.model.fit(self.movies_data, self.ratings_data)
        
        # Тест предсказаний для существующего пользователя
        predictions = self.model.predict(user_id=1, top_n=2)
        self.assertEqual(len(predictions), 2)
        self.assertTrue(all(isinstance(pred[0], int) for pred in predictions))
        self.assertTrue(all(isinstance(pred[1], float) for pred in predictions))
        
        # Тест предсказаний для несуществующего пользователя
        with self.assertRaises(ValueError):
            self.model.predict(user_id=999)

    def test_alpha_weighting_effect(self):
        model_0 = HybridModel(alpha=0.0)
        model_0.fit(self.movies_data, self.ratings_data)
        result_0 = model_0.predict(user_id=1, top_n=5)

        model_1 = HybridModel(alpha=1.0)
        model_1.fit(self.movies_data, self.ratings_data)
        result_1 = model_1.predict(user_id=1, top_n=5)

        self.assertNotEqual(result_0, result_1, msg="Рекомендации не изменились при смене alpha")

    def test_predict_ranks_content_vs_collab(self):
        """
        Проверяет, что при alpha=1.0 рекомендации совпадают с контентной моделью,
        а при alpha=0.0 - с коллаборативной моделью.
        """
        self.model.fit(self.movies_data, self.ratings_data)

        # alpha = 1 → должна совпадать с content-based
        pure_content = HybridModel(alpha=1.0)
        pure_content.fit(self.movies_data, self.ratings_data)
        content_preds = pure_content.predict(user_id=1, top_n=5)

        # alpha = 0 → должна совпадать с collaborative
        pure_collab = HybridModel(alpha=0.0)
        pure_collab.fit(self.movies_data, self.ratings_data)
        collab_preds = pure_collab.predict(user_id=1, top_n=5)

        # Сравним с тем, что выдают внутренние модели напрямую
        internal_content_preds = self.model.content_model.predict(user_id=1, top_n=5)
        
        # Нормализуем коллаборативные рекомендации так же, как в гибридной модели
        internal_collab_preds = self.model.collaborative_model.predict(user_id=1, top_n=5)
        internal_collab_preds = [(mid, (score - 0.5) / 4.5) for mid, score in internal_collab_preds]

        # Сравниваем и ID фильмов, и их scores
        self.assertEqual(content_preds, internal_content_preds,
                        msg="Content-based predictions mismatch with alpha=1.0")

        self.assertEqual(collab_preds, internal_collab_preds,
                        msg="Collaborative predictions mismatch with alpha=0.0")
        
    def test_empty_user_behavior(self):
        self.model.fit(self.movies_data, self.ratings_data)

        # Добавим нового пользователя, которого нет в ratings
        new_user_id = 999
        with self.assertRaises(ValueError):
            self.model.predict(user_id=new_user_id)


if __name__ == '__main__':
    unittest.main() 