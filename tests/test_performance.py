import pandas as pd
import numpy as np
import time
import psutil
import os
import sys
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Добавляем корневую директорию проекта в PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.content_based import ContentBasedModel
from models.collaborative import CollaborativeModel
from models.hybrid import HybridModel

def get_memory_usage() -> float:
    """Возвращает текущее использование памяти в МБ"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_cpu_percent() -> float:
    """Возвращает текущую загрузку CPU в процентах"""
    return psutil.cpu_percent()

def measure_training_performance(model_name: str, 
                               model, 
                               movies_df: pd.DataFrame, 
                               train_ratings: pd.DataFrame) -> Dict:
    """
    Измеряет производительность при обучении модели.
    
    Returns:
        Словарь с метриками производительности:
        - training_time: время обучения в секундах
        - memory_usage: использование памяти в МБ
        - avg_cpu: средняя загрузка CPU
        - max_cpu: максимальная загрузка CPU
    """
    print(f"\nИзмерение производительности обучения модели {model_name}...")
    
    # Начальные значения
    start_memory = get_memory_usage()
    cpu_percentages = []
    
    # Запускаем мониторинг CPU в отдельном потоке
    def monitor_cpu():
        while True:
            cpu_percentages.append(get_cpu_percent())
            time.sleep(0.1)
    
    import threading
    monitor_thread = threading.Thread(target=monitor_cpu, daemon=True)
    monitor_thread.start()
    
    # Замеряем время обучения
    start_time = time.time()
    
    try:
        if model_name == 'Collaborative':
            model.fit(train_ratings)
        else:
            model.fit(movies_df, train_ratings)
    except Exception as e:
        print(f"Ошибка при обучении модели {model_name}: {str(e)}")
        return None
    
    training_time = time.time() - start_time
    
    # Останавливаем мониторинг CPU
    monitor_thread.join(timeout=0.1)
    
    # Финальные значения
    end_memory = get_memory_usage()
    memory_usage = end_memory - start_memory
    
    return {
        'training_time': training_time,
        'memory_usage': memory_usage,
        'avg_cpu': np.mean(cpu_percentages),
        'max_cpu': np.max(cpu_percentages)
    }

def measure_prediction_performance(model_name: str, 
                                 model, 
                                 test_ratings: pd.DataFrame,
                                 n_predictions: int = 100) -> Dict:
    """
    Измеряет производительность при выдаче рекомендаций.
    
    Args:
        n_predictions: количество пользователей для тестирования
        
    Returns:
        Словарь с метриками производительности:
        - avg_prediction_time: среднее время выдачи рекомендаций в секундах
        - max_prediction_time: максимальное время выдачи рекомендаций
    """
    print(f"\nИзмерение производительности выдачи рекомендаций для модели {model_name}...")
    
    # Выбираем случайных пользователей для тестирования
    test_users = np.random.choice(
        test_ratings['userId'].unique(),
        size=min(n_predictions, len(test_ratings['userId'].unique())),
        replace=False
    )
    
    prediction_times = []
    
    for user_id in test_users:
        start_time = time.time()
        try:
            model.predict(user_id, top_n=10)
            prediction_time = time.time() - start_time
            prediction_times.append(prediction_time)
        except Exception as e:
            print(f"Ошибка при получении рекомендаций для пользователя {user_id}: {str(e)}")
            continue
    
    return {
        'avg_prediction_time': np.mean(prediction_times),
        'max_prediction_time': np.max(prediction_times)
    }

def plot_performance_results(results: Dict):
    """Визуализирует результаты тестов производительности"""
    # Подготовка данных для графиков
    models = list(results.keys())
    metrics = ['training_time', 'memory_usage', 'avg_cpu', 'max_cpu', 'avg_prediction_time']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Результаты тестов производительности')
    
    # Графики для каждой метрики
    for i, metric in enumerate(metrics):
        row = i // 3
        col = i % 3
        
        values = [results[model][metric] for model in models]
        
        axes[row, col].bar(models, values)
        axes[row, col].set_title(metric)
        axes[row, col].tick_params(axis='x', rotation=45)
        
        # Добавляем значения над столбцами
        for j, v in enumerate(values):
            axes[row, col].text(j, v, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('performance_results.png')
    plt.close()

def main():
    print("Загрузка данных...")
    movies_df = pd.read_parquet('data/raw/movies_small.parquet')
    ratings_df = pd.read_parquet('data/raw/ratings_small.parquet')
    
    # Инициализируем модели
    models = {
        'Content-Based': ContentBasedModel(),
        'Collaborative': CollaborativeModel(),
        'Hybrid': HybridModel()
    }
    
    # Результаты тестов производительности
    performance_results = {}
    
    # Тестируем каждую модель
    for model_name, model in models.items():
        print(f"\nТестирование модели: {model_name}")
        print("-" * 50)
        
        # Измеряем производительность обучения
        training_metrics = measure_training_performance(
            model_name, model, movies_df, ratings_df
        )
        
        if training_metrics is None:
            continue
        
        # Измеряем производительность выдачи рекомендаций
        prediction_metrics = measure_prediction_performance(
            model_name, model, ratings_df
        )
        
        # Объединяем результаты
        performance_results[model_name] = {
            **training_metrics,
            **prediction_metrics
        }
        
        # Выводим результаты
        print(f"\nРезультаты для модели {model_name}:")
        print(f"Время обучения: {training_metrics['training_time']:.2f} сек")
        print(f"Использование памяти: {training_metrics['memory_usage']:.2f} МБ")
        print(f"Средняя загрузка CPU: {training_metrics['avg_cpu']:.2f}%")
        print(f"Максимальная загрузка CPU: {training_metrics['max_cpu']:.2f}%")
        print(f"Среднее время выдачи рекомендаций: {prediction_metrics['avg_prediction_time']:.4f} сек")
        print(f"Максимальное время выдачи рекомендаций: {prediction_metrics['max_prediction_time']:.4f} сек")
    
    # Сохраняем результаты в CSV
    results_df = pd.DataFrame(performance_results).T
    results_df.to_csv('performance_results.csv')
    print("\nРезультаты сохранены в performance_results.csv")
    
    # Создаем визуализацию
    plot_performance_results(performance_results)
    print("Графики сохранены в performance_results.png")

if __name__ == "__main__":
    main() 