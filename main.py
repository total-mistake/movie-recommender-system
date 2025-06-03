import time
import os
from memory_profiler import memory_usage
from psutil import Process
from config import RATINGS_DATA_PATH, MOVIE_METADATA_PATH, CONTENT_MODEL_PATH
import pandas as pd
from models.hybrid import HybridModel
from models.content_based import ContentBasedModel
from models.collaborative import CollaborativeModel
from database.connection import get_movies_data, get_ratings_data
import logging

# Настройка логирования для всех модулей
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_model():
    logger.info("Loading data...")
    movies = get_movies_data()
    ratings = get_ratings_data()

    # Оставляем только нужные колонки
    movies = movies[['Movie_ID', 'Type', 'Plot', 'Year', 'Genres', 'Directors', 'Writers', 'Actors', 'Countries']]
    
    # Если есть старая модель, удаляем её
    if os.path.exists(CONTENT_MODEL_PATH):
        logger.info(f"Removing old model at {CONTENT_MODEL_PATH}")
        os.remove(CONTENT_MODEL_PATH)
    
    logger.info("Training new model...")
    model = ContentBasedModel()
    model.fit(movies, ratings)
    
    # Проверяем, что модель сохранилась
    if os.path.exists(CONTENT_MODEL_PATH):
        logger.info("Model saved successfully")
        
        # Проверяем, что модель загружается
        logger.info("Testing model loading...")
        test_model = ContentBasedModel()
        
        # Проверяем добавление пользователя
        logger.info("Testing user addition...")
        test_model.add_new_user(100000, ['Action', 'Adventure', 'Fantasy'])
        
        # Проверяем получение рекомендаций
        logger.info("Testing recommendations...")
        recommendations = test_model.predict(100000, 10)
        logger.info(f"Recommendations: {recommendations}")
        
        # Проверяем похожие фильмы
        logger.info("Testing similar movies...")
        similar = test_model.get_similar_movies(1, 10)
        logger.info(f"Similar movies: {similar}")
    else:
        logger.error("Failed to save model!")

if __name__ == "__main__":
    model = ContentBasedModel()
    movie = get_movies_data().iloc[0]
    movie['Movie_ID'] = 84999
    print(movie)
    model.add_movie(movie)
    print(model.get_similar_movies(84999, 10))
