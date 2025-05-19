# Пути к данным
RATINGS_DATA_PATH = "data/raw/ratings.parquet"
MOVIE_METADATA_PATH = "data/raw/movies.parquet"
COLLABORATIVE_MODEL_PATH = "data/processed/collaborative_model.pkl"
CONTENT_MODEL_PATH = "data/processed/content_model.pkl"
HYBRID_MODEL_PATH = "data/processed/hybrid_model.pkl"

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