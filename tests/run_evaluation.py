import pandas as pd
import sys
import os
import numpy as np
import time

# Добавляем корневую директорию проекта в PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.content_based import ContentBasedModel
from models.collaborative import CollaborativeModel
from models.hybrid import HybridModel
from test_evaluation import evaluate_model

# Константы для оценки моделей
N_VALUES = [10, 50, 100]  # Значения n для метрик precision@n, recall@n, ndcg@n

def load_data():
    """
    Загружает данные и разделяет их на обучающую и тестовую выборки.
    Для каждого пользователя последние 20% его оценок идут в test, остальные в train.
    Гарантирует, что каждый пользователь будет представлен в обеих выборках.
    """
    print("Загрузка данных...")
    
    # Загружаем данные
    movies_df = pd.read_parquet('data/raw/movies_small.parquet')
    ratings_df = pd.read_parquet('data/raw/ratings_small.parquet')
    
    # Сортируем рейтинги по времени для каждого пользователя
    ratings_df = ratings_df.sort_values(['userId', 'date'])
    
    # Сбрасываем индексы для корректной работы с масками
    ratings_df = ratings_df.reset_index(drop=True)
    
    # Создаем маску для разделения на train/test
    train_mask = np.ones(len(ratings_df), dtype=bool)
    
    # Для каждого пользователя берем последние 20% оценок в test
    for user_id, group in ratings_df.groupby('userId'):
        user_indices = group.index
        n_test = int(len(user_indices) * 0.2)
        if n_test > 0:  # если у пользователя достаточно оценок
            test_indices = user_indices[-n_test:]  # берем последние n_test индексов
            train_mask[test_indices] = False
    
    # Применяем маску для разделения
    train_ratings = ratings_df[train_mask]
    test_ratings = ratings_df[~train_mask]
    
    print(f"Размер обучающей выборки: {len(train_ratings)}")
    print(f"Размер тестовой выборки: {len(test_ratings)}")
    print(f"Количество пользователей в тренировочной выборке: {len(train_ratings['userId'].unique())}")
    print(f"Количество пользователей в тестовой выборке: {len(test_ratings['userId'].unique())}")
    print(f"Количество пользователей в исходном датасете: {len(ratings_df['userId'].unique())}")
    
    # Проверяем, что все пользователи есть в обеих выборках
    train_users = set(train_ratings['userId'].unique())
    test_users = set(test_ratings['userId'].unique())
    all_users = set(ratings_df['userId'].unique())
    print(f"Все пользователи в train: {train_users == all_users}")
    print(f"Все пользователи в test: {test_users == all_users}")
    
    return movies_df, train_ratings, test_ratings

def train_and_evaluate_models(movies_df, train_ratings, test_ratings):
    """
    Обучает и оценивает все модели рекомендаций.
    """
    
    results = {}
    
    # Сначала обучаем базовые модели
    print("\nОценка модели: Content-Based")
    print("-" * 50)
    content_model = ContentBasedModel()
    content_model.fit(movies_df, train_ratings)
    content_results = evaluate_model(
        model=content_model,
        test_ratings=test_ratings,
        n_values=N_VALUES
    )
    results['Content-Based'] = content_results
    
    print("\nОценка модели: Collaborative")
    print("-" * 50)
    collab_model = CollaborativeModel()
    collab_model.fit(train_ratings)
    collab_results = evaluate_model(
        model=collab_model,
        test_ratings=test_ratings,
        n_values=N_VALUES
    )
    results['Collaborative'] = collab_results
    
    # Теперь создаем и оцениваем гибридные модели
    hybrid_alphas = [0.2, 0.4, 0.6, 0.8]
    for alpha in hybrid_alphas:
        model_name = f'Hybrid-{alpha}'
        print(f"\nОценка модели: {model_name}")
        print("-" * 50)
        
        try:
            hybrid_model = HybridModel(alpha=alpha)
            # Гибридная модель использует уже обученные базовые модели
            hybrid_results = evaluate_model(
                model=hybrid_model,
                test_ratings=test_ratings,
                n_values=N_VALUES
            )
            results[model_name] = hybrid_results
            
            # Выводим результаты
            print("\nРезультаты:")
            for metric, scores in hybrid_results.items():
                print(f"\n{metric.upper()}:")
                for n, score in scores.items():
                    print(f"@{n}: {score:.4f}")
                    
        except Exception as e:
            print(f"Ошибка при оценке модели {model_name}: {str(e)}")
            continue
    
    return results

def reduce_dataset_size(ratings_df, movies_df, user_fraction=0.1, random_state=42):
    """
    Уменьшает размер датасета, удаляя случайную долю пользователей и все их рейтинги.
    
    Args:
        ratings_df: DataFrame с рейтингами
        movies_df: DataFrame с информацией о фильмах
        user_fraction: Доля пользователей для удаления (от 0 до 1)
        random_state: Seed для воспроизводимости
        
    Returns:
        ratings_df_reduced, movies_df_reduced
    """
    print(f"\nУдаляем {user_fraction*100}% пользователей из датасета...")
    
    # Получаем список всех пользователей
    all_users = ratings_df['userId'].unique()
    
    # Выбираем случайную подвыборку пользователей для удаления
    np.random.seed(random_state)
    users_to_remove = np.random.choice(
        all_users, 
        size=int(len(all_users) * user_fraction), 
        replace=False
    )
    
    # Удаляем выбранных пользователей и их рейтинги
    ratings_df_reduced = ratings_df[~ratings_df['userId'].isin(users_to_remove)]
    
    # Получаем список оставшихся фильмов
    remaining_movies = ratings_df_reduced['movieId'].unique()
    movies_df_reduced = movies_df[movies_df['movieId'].isin(remaining_movies)]
    
    print(f"Исходный размер датасета:")
    print(f"- Количество пользователей: {len(all_users)}")
    print(f"- Количество рейтингов: {len(ratings_df)}")
    print(f"- Количество фильмов: {len(movies_df)}")
    
    print(f"\nУменьшенный размер датасета:")
    print(f"- Количество пользователей: {len(ratings_df_reduced['userId'].unique())}")
    print(f"- Количество рейтингов: {len(ratings_df_reduced)}")
    print(f"- Количество фильмов: {len(movies_df_reduced)}")
    
    return ratings_df_reduced, movies_df_reduced

def main():
    start_time = time.time()
    
    # # Загружаем данные
    # movies_df = pd.read_parquet('data/raw/movies_test.parquet')
    # ratings_df = pd.read_parquet('data/raw/ratings_test.parquet')
    
    # # Уменьшаем размер датасета (удаляем 90% пользователей)
    # ratings_df, movies_df = reduce_dataset_size(ratings_df, movies_df, user_fraction=0.9)

    # ratings_df.to_parquet('data/raw/ratings_small.parquet')
    # movies_df.to_parquet('data/raw/movies_small.parquet')
    
    # Разделяем на train/test
    movies_df, train_ratings, test_ratings = load_data()
    
    # Обучаем и оцениваем модели
    results = train_and_evaluate_models(movies_df, train_ratings, test_ratings)
    
    # Сохраняем результаты в файл
    results_df = pd.DataFrame()
    
    for model_name, model_results in results.items():
        for metric, scores in model_results.items():
            for n, score in scores.items():
                results_df = pd.concat([results_df, pd.DataFrame({
                    'model': [model_name],
                    'metric': [metric],
                    'n': [n],
                    'score': [score]
                })])
    
    results_df.to_csv('evaluation_results.csv', index=False)
    print("\nРезультаты сохранены в evaluation_results.csv")
    
    end_time = time.time()
    print(f"Время выполнения: {end_time - start_time:.2f} секунд")

if __name__ == "__main__":
    main() 