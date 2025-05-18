import time
import os
from surprise.model_selection import KFold, LeaveOneOut, cross_validate
from surprise import Dataset, Reader, accuracy
from collections import defaultdict
from memory_profiler import memory_usage
from psutil import Process


def test_kfold(model, dataset, n_splits=5):
    kf = KFold(n_splits=n_splits, random_state=9, shuffle=True)
    start = time.time()
    results = cross_validate(model, dataset, measures=["RMSE", "MAE"], cv=kf, verbose=True,  n_jobs=-1)
    end = time.time()
    print(f"[INFO] Средний RMSE: {results['test_rmse'].mean():.4f}")
    print(f"[INFO] Средний MAE: {results['test_mae'].mean():.4f}")
    print(f"[INFO] Время выполнения K-Fold: {end - start:.2f} секунд")
    return results


def test_leave_one_out(model, ratings_df):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], reader)
    loo = LeaveOneOut(n_splits=1, random_state=42)

    for trainset, testset in loo.split(data):
        model.fit(trainset)
        predictions = model.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        return rmse, mae


def profile_memory_time(model, trainset):
    start = time.time()
    proc = Process(os.getpid())
    print(f"Пиковое использование памяти: {max(memory_usage((model.fit, (trainset,)))):.2f} MiB")
    mem_info = proc.memory_info()
    print(f"Текущее использование памяти: {mem_info.rss / (1024 ** 2):.2f} MiB")
    print(f"[INFO] Время выполнения: {time.time() - start:.2f} секунд")


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions, recalls = {}, {}
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in top_k)
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold)) for (est, true_r) in top_k
        )

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel else 0

    return precisions, recalls
