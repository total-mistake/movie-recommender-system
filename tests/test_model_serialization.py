import unittest
import os
import sys
import time
import pickle
import joblib
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.content_based import ContentBasedModel
from config import CONTENT_MODEL_PATH

class TestModelSerialization(unittest.TestCase):
    """Тесты для сравнения производительности сериализации моделей"""
    
    def setUp(self):
        """Подготовка тестовых данных"""
        # Загружаем существующую модель
        self.model = ContentBasedModel()
        if not self.model.model_exists():
            raise RuntimeError("Модель не найдена. Сначала обучите модель.")
        
        # Создаем временные пути для сохранения
        self.pickle_path = "data/processed/temp_model.pkl"
        self.joblib_path = "data/processed/temp_model.joblib"
        
        # Создаем директорию если её нет
        os.makedirs(os.path.dirname(self.pickle_path), exist_ok=True)
    
    def tearDown(self):
        """Очистка после тестов"""
        # Удаляем временные файлы
        if os.path.exists(self.pickle_path):
            os.remove(self.pickle_path)
        if os.path.exists(self.joblib_path):
            os.remove(self.joblib_path)
    
    def test_serialization_performance(self):
        """Сравнение производительности pickle и joblib"""
        # Подготовка данных для сохранения
        model_data = {
            'movie_ids': self.model.movie_ids,
            'feature_matrix': self.model.feature_matrix,
            'user_profiles': self.model.user_profiles,
            'watched_movies': self.model.watched_movies,
            'preprocessor': self.model.preprocessor
        }

        print('Сохраняем модель в pickle')
        
        # Тест pickle
        pickle_save_start = time.time()
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(model_data, f)
        pickle_save_time = time.time() - pickle_save_start
        print(f'Pickle сохранение: {pickle_save_time:.3f} сек')
        
        pickle_load_start = time.time()
        with open(self.pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)
        pickle_load_time = time.time() - pickle_load_start
        print(f'Pickle загрузка: {pickle_load_time:.3f} сек')
        
        # Тест joblib
        joblib_save_start = time.time()
        joblib.dump(model_data, self.joblib_path)
        joblib_save_time = time.time() - joblib_save_start
        print(f'Joblib сохранение: {joblib_save_time:.3f} сек')
        
        joblib_load_start = time.time()
        joblib_data = joblib.load(self.joblib_path)
        joblib_load_time = time.time() - joblib_load_start
        print(f'Joblib загрузка: {joblib_load_time:.3f} сек')
        

        # Вывод результатов
        print("\nРезультаты сравнения сериализации:")
        print(f"Pickle сохранение: {pickle_save_time:.3f} сек")
        print(f"Pickle загрузка: {pickle_load_time:.3f} сек")
        print(f"Joblib сохранение: {joblib_save_time:.3f} сек")
        print(f"Joblib загрузка: {joblib_load_time:.3f} сек")
        
        # Проверяем корректность загрузки
        self.assertEqual(len(pickle_data['movie_ids']), len(joblib_data['movie_ids']))
        self.assertEqual(pickle_data['movie_ids'].shape, joblib_data['movie_ids'].shape)
        self.assertEqual(pickle_data['feature_matrix'].shape, joblib_data['feature_matrix'].shape)
        self.assertEqual(len(pickle_data['user_profiles']), len(joblib_data['user_profiles']))
        self.assertEqual(len(pickle_data['watched_movies']), len(joblib_data['watched_movies']))

if __name__ == '__main__':
    unittest.main() 