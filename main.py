from models.collaborative import CollaborativeModel
import time
from config import RAW_DATA_PATH
import pandas as pd

model = CollaborativeModel()

if model.model is None:
    ratings_df = pd.read_parquet(RAW_DATA_PATH)
    model.fit(ratings_df)

start = time.time()
slow = model.predict(1)
slow_time = time.time() - start
print(f"Time: {slow_time:.4f} sec\n")
