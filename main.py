import time
import os
from memory_profiler import memory_usage
from psutil import Process
from config import RATINGS_DATA_PATH, MOVIE_METADATA_PATH
import pandas as pd
from models.hybrid import HybridModel
from models.content_based import ContentBasedModel
from models.collaborative import CollaborativeModel

model = HybridModel()
movies = pd.read_parquet(MOVIE_METADATA_PATH)
ratings = pd.read_parquet(RATINGS_DATA_PATH)
start = time.time()
rec = model.predict(635, top_n=10)
end = time.time()
print(f"Время выполнения: {end - start:.2f} секунд")

# content_model = ContentBasedModel()
# movies = pd.read_parquet(MOVIE_METADATA_PATH)
# ratings = pd.read_parquet(RATINGS_DATA_PATH)
# start = time.time()
# content_recs = dict(content_model.predict(635, top_n=None))
# end = time.time()
# print(f"Время выполнения: {end - start:.2f} секунд")
# start = time.time()
# content_recs = dict(content_model.predict(635, top_n=None))
# end = time.time()
# print(f"Время выполнения: {end - start:.2f} секунд")

# start = time.time()
# model = CollaborativeModel()
# print(f"[INFO] Модель загружена. {time.time() - start:.2f} секунд")
# start = time.time()
# rec2 = model.predict_fast(635, top_n=10)
# print(f"Время выполнения: {time.time() - start:.2f} секунд")
# start = time.time()
# rec = model.predict(635, top_n=10)
# print(f"Время выполнения: {time.time() - start:.2f} секунд")
