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
movie = {
    "movieId": 292758,
    "title": "Final Destination: Bloodlines",
    "plot": "Plagued by a recurring violent nightmare, a college student returns home to find the one person who can break the cycle and save her family from the horrific fate that inevitably awaits them.",
    "genres": "Horror",
    "directors": "Zach LipovskyAdam, B. Stein",
    "writers": "Guy Busick, Lori Evans Taylor, Jon Watts, Guy Busick, Lori Evans Taylor",
    "actors": "Kaitlyn Santa Juana, Stefani Reyes, Teo Briones, Teo Briones, Charlie Reyes",
    "countries": "United States",
    "start_year": 2025,
    "type": "movie"
}
model.content_model.add_movie(movie)
print(model.content_model.get_similar_movies(292758, top_n=10))


# start = time.time()
# model.content_model.recommend_recent(635, top_n=10)
# end = time.time()
# print(f"Время выполнения: {end - start:.2f} секунд")

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
# rec2 = model.predict(635, top_n=10)
# print(f"Время выполнения: {time.time() - start:.2f} секунд")
# start = time.time()
# rec = model.predict_(635, top_n=10)
# print(f"Время выполнения: {time.time() - start:.2f} секунд")
# start = time.time()
# rec2 = model.predict(635, top_n=10)
# print(f"Время выполнения: {time.time() - start:.2f} секунд")
# start = time.time()
# rec = model.predict_(635, top_n=10)
# print(f"Время выполнения: {time.time() - start:.2f} секунд")
