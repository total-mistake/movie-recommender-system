# Пути к данным
RATINGS_DATA_PATH = "data/raw/ratings.parquet"
MOVIE_METADATA_PATH = "data/raw/movies.parquet"
COLLABORATIVE_MODEL_PATH = "data/processed/collaborative_model.pkl"
CONTENT_MODEL_PATH = "data/processed/content_model.pkl"
HYBRID_MODEL_PATH = "data/processed/hybrid_model.pkl"
COLLABORATIVE_TEST_MODEL_PATH = "data/processed/collaborative_test_model.pkl"
CONTENT_TEST_MODEL_PATH = "data/processed/content_test_model.pkl"
HYBRID_TEST_MODEL_PATH = "data/processed/hybrid_test_model.pkl"

# Параметры моделей
SVD_PARAMS = {
    "n_factors": 100,
    'biased': True,
    "n_epochs": 40,
    "lr_all": 0.005,
    "reg_all": 0.05
}

# Гибридизация
HYBRID_ALPHA = 0.6  # Доля контентной модели

# Конфигурация базы данных
DB_USERNAME = 'root'
DB_PASSWORD = '314159'
DB_HOST = 'localhost'
DB_PORT = 3306
DB_NAME = 'movies_db'
DATABASE_URL = f'mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

# Конфигурация API
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Movie Recommender API"
API_DESCRIPTION = "API для системы рекомендации фильмов"
API_VERSION = "1.0.0"

# Конфигурация JWT
JWT_SECRET_KEY = "your-secret-key"  # В продакшене использовать безопасный ключ
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Конфигурация IMDb API
IMDB_API_URL = "https://graph.imdbapi.dev/v1"
IMDB_API_TIMEOUT = 10

# Конфигурация CORS
CORS_ORIGINS = ["*"]  # В продакшене заменить на конкретные домены
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]