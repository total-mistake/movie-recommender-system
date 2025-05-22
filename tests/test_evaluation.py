import numpy as np
import pandas as pd
import time
from typing import List, Tuple, Dict, Any

def calculate_precision_at_n(recommended_items: List[Tuple[int, float]], 
                           relevant_items: List[int], 
                           n: int) -> float:
    """
    Вычисляет Precision@N для рекомендаций.
    
    Precision@N - это доля релевантных элементов среди первых N рекомендаций.
    Например, если из 10 рекомендаций 3 оказались релевантными, то Precision@10 = 0.3
    
    Args:
        recommended_items: Список кортежей (id_фильма, score) отсортированных по score
        relevant_items: Список id релевантных фильмов
        n: Количество первых рекомендаций для оценки
        
    Returns:
        Значение метрики Precision@N
    """
    if n == 0:
        return 0.0
    
    top_n_items = [item[0] for item in recommended_items[:n]]
    hits = sum(1 for item in top_n_items if item in relevant_items)
    return hits / n

def calculate_recall_at_n(recommended_items: List[Tuple[int, float]], 
                         relevant_items: List[int], 
                         n: int) -> float:
    """
    Вычисляет Recall@N для рекомендаций.
    
    Recall@N - это доля найденных релевантных элементов среди всех релевантных.
    Например, если у пользователя 10 релевантных фильмов, и мы нашли 3 из них
    в первых 20 рекомендациях, то Recall@20 = 0.3
    
    Args:
        recommended_items: Список кортежей (id_фильма, score) отсортированных по score
        relevant_items: Список id релевантных фильмов
        n: Количество первых рекомендаций для оценки
        
    Returns:
        Значение метрики Recall@N
    """
    if not relevant_items:
        return 0.0
    
    top_n_items = [item[0] for item in recommended_items[:n]]
    hits = sum(1 for item in top_n_items if item in relevant_items)
    return hits / len(relevant_items)

def calculate_ndcg_at_n(recommended_items: List[Tuple[int, float]], 
                       relevant_items: List[int], 
                       n: int) -> float:
    """
    Вычисляет NDCG@N (Normalized Discounted Cumulative Gain) для рекомендаций.
    
    NDCG учитывает позицию релевантных элементов в списке рекомендаций.
    Чем выше позиция релевантного элемента, тем больше его вклад в итоговую оценку.
    
    Args:
        recommended_items: Список кортежей (id_фильма, score) отсортированных по score
        relevant_items: Список id релевантных фильмов
        n: Количество первых рекомендаций для оценки
        
    Returns:
        Значение метрики NDCG@N
    """
    if not relevant_items:
        return 0.0
    
    # Создаем список релевантности (1 для релевантных элементов, 0 для остальных)
    relevance_scores = [1.0 if item[0] in relevant_items else 0.0 
                       for item in recommended_items[:n]]
    
    # Вычисляем DCG (Discounted Cumulative Gain)
    dcg = sum((2**score - 1) / np.log2(i + 2) 
              for i, score in enumerate(relevance_scores))
    
    # Вычисляем IDCG (Ideal DCG) - идеальный случай, когда все релевантные элементы
    # находятся в начале списка
    ideal_scores = [1.0] * min(len(relevant_items), n)
    idcg = sum((2**score - 1) / np.log2(i + 2) 
               for i, score in enumerate(ideal_scores))
    
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_model(model: Any, 
                  test_ratings: pd.DataFrame,
                  n_values: List[int] = [5, 10, 20]) -> Dict[str, Dict[int, float]]:
    """
    Оценивает качество рекомендаций модели по нескольким метрикам.
    
    Args:
        model: Экземпляр модели рекомендаций
        test_ratings: DataFrame с тестовыми рейтингами (userId, movieId, rating)
        n_values: Список значений N для расчета метрик
        
    Returns:
        Словарь с результатами метрик для каждого N
    """
    results = {
        'precision': {n: [] for n in n_values},
        'recall': {n: [] for n in n_values},
        'ndcg': {n: [] for n in n_values}
    }
    
    # Группируем рейтинги по пользователям для оптимизации
    user_ratings = test_ratings.groupby('userId')
    total_users = len(user_ratings)
    processed_users = 0

    # Предварительно получаем все рекомендации для всех пользователей
    all_recommendations = {}
    for user_id, user_data in user_ratings:
        try:
            recommendations = model.predict(user_id, top_n=max(n_values))
            all_recommendations[user_id] = recommendations
            processed_users += 1
            if processed_users % 1000 == 0:
                print(f"Обработано пользователей: {processed_users}/{total_users}")
        except Exception as e:
            print(f"Ошибка при получении рекомендаций для пользователя {user_id}: {str(e)}")
            continue
    
    # Создаем множества релевантных фильмов для каждого пользователя
    user_relevant_movies = {
        user_id: set(user_data['movieId'].tolist())
        for user_id, user_data in user_ratings
    }
    
    # Вычисляем метрики для каждого пользователя
    for user_id, recommendations in all_recommendations.items():
        relevant_movies = user_relevant_movies[user_id]
        
        # Вычисляем метрики для каждого N
        for n in n_values:
            results['precision'][n].append(
                calculate_precision_at_n(recommendations, relevant_movies, n)
            )
            results['recall'][n].append(
                calculate_recall_at_n(recommendations, relevant_movies, n)
            )
            results['ndcg'][n].append(
                calculate_ndcg_at_n(recommendations, relevant_movies, n)
            )

    # Вычисляем средние значения метрик
    for metric in results:
        for n in n_values:
            scores = results[metric][n]
            results[metric][n] = np.mean(scores) if scores else 0.0
            
    return results

def test_evaluation_metrics():
    """Тестирует вычисление метрик на простых примерах"""
    # Тестовые данные
    recommended = [(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6), (5, 0.5)]
    relevant = [1, 3, 5]
    
    # Тест Precision@N
    assert abs(calculate_precision_at_n(recommended, relevant, 3) - 0.67) < 0.01
    assert abs(calculate_precision_at_n(recommended, relevant, 5) - 0.6) < 0.01
    
    # Тест Recall@N
    assert abs(calculate_recall_at_n(recommended, relevant, 3) - 0.67) < 0.01
    assert abs(calculate_recall_at_n(recommended, relevant, 5) - 1.0) < 0.01
    
    # Тест NDCG@N
    assert calculate_ndcg_at_n(recommended, relevant, 3) > 0
    assert calculate_ndcg_at_n(recommended, relevant, 5) > 0
    
