import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from surprise import Dataset, Reader
from config import RAW_DATA_PATH
from models.collaborative import CollaborativeModel
from utils import (
    test_kfold,
    test_leave_one_out,
    measure_train_time,
    profile_memory_time,
    precision_recall_at_k
)

if __name__ == "__main__":
    # Инициализация модели
    model = CollaborativeModel()
    print("Запуск теста на память и время")
        
    # Загружаем данные
    ratings_df = pd.read_parquet(RAW_DATA_PATH)

    profile_memory_time(model, ratings_df)

    # Загружаем данные
    ratings_df = pd.read_parquet(RAW_DATA_PATH)
    reader = Reader(rating_scale=(ratings_df.rating.min(), ratings_df.rating.max()))
    data = Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], reader)
    trainset = data.build_full_trainset()

    # 1. Кросс-валидация
    print("== K-Fold Validation ==")
    results = test_kfold(model, data)
    print(results)

    # # 2. Leave-One-Out
    # print("\n== Leave-One-Out ==")
    # rmse, mae = test_leave_one_out(model, ratings_df)
    # print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # # 5. Precision/Recall
    # print("\n== Precision / Recall ==")
    # # Получим предсказания и рассчитаем метрики
    # predictions = model._model.test(trainset.build_testset())
    # precisions, recalls = precision_recall_at_k(predictions)
    # avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
    # avg_recall = sum(rec for rec in recalls.values()) / len(recalls)
    # print(f"Average Precision@10: {avg_precision:.4f}")
    # print(f"Average Recall@10: {avg_recall:.4f}")


