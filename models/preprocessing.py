from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# Классы для строковых импутеров и токенизации
class SplitList(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [[item.strip() for item in (entry or '').split(',') if item and item.strip()] for entry in X]

class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer(sparse_output=True)

    def fit(self, X, y=None):
        self.mlb.fit(X)
        return self

    def transform(self, X):
        return self.mlb.transform(X)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Предобработка текстовых данных с обработкой пустых значений"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Заменяем None и NaN на пустую строку
        return [str(x) if x is not None else '' for x in X]

# Функции
def reshape_column(X): return X.values.reshape(-1, 1)
def boost_year(X): return X * 3.0
def scale_half(X): return X * 0.5

# Пайплайны
def make_numeric_pipeline():
    return Pipeline([
        ('reshape', FunctionTransformer(reshape_column, validate=False)),
        ('impute', SimpleImputer(strategy='constant', fill_value=0)),
        ('scale', StandardScaler()),
        ('boost', FunctionTransformer(boost_year, validate=False))
    ])

class DynamicLSA(BaseEstimator, TransformerMixin):
    """
    LSA с динамическим выбором количества компонент на основе объясненной дисперсии.
    
    Args:
        variance_threshold (float): Минимальный порог объясненной дисперсии (0-1).
            По умолчанию 0.95 (95% информации).
        max_components (int): Максимальное количество компонент.
            По умолчанию 300.
        min_components (int): Минимальное количество компонент.
            По умолчанию 10.
    """
    def __init__(self, variance_threshold=0.95, max_components=300, min_components=10):
        self.variance_threshold = variance_threshold
        self.max_components = max_components
        self.min_components = min_components
        self.n_components_ = None
        self.svd_ = None
        self.explained_variance_ratio_ = None
        
    def fit(self, X, y=None):
        # Определяем максимально возможное количество компонент
        n_samples, n_features = X.shape
        max_possible_components = min(n_samples, n_features, self.max_components)
        
        if max_possible_components < self.min_components:
            raise ValueError(
                f"Размерность данных ({n_samples}, {n_features}) слишком мала для "
                f"заданного min_components={self.min_components}"
            )
        
        # Сначала пробуем с максимально возможным количеством компонент
        self.svd_ = TruncatedSVD(n_components=max_possible_components)
        self.svd_.fit(X)
        self.explained_variance_ratio_ = self.svd_.explained_variance_ratio_
        
        # Находим минимальное количество компонент, объясняющих нужную долю дисперсии
        cumulative_variance = np.cumsum(self.explained_variance_ratio_)
        
        # Если не удается достичь порога, используем максимально возможное количество
        if cumulative_variance[-1] < self.variance_threshold:
            print(f"[WARNING] Не удалось достичь порога дисперсии {self.variance_threshold}. "
                  f"Максимальная достижимая дисперсия: {cumulative_variance[-1]:.3f}")
            self.n_components_ = max_possible_components
        else:
            self.n_components_ = np.argmax(cumulative_variance >= self.variance_threshold) + 1
            # Ограничиваем снизу минимальным количеством компонент
            self.n_components_ = max(self.n_components_, self.min_components)
        
        print(f"[INFO] Выбрано {self.n_components_} компонент, "
              f"объясняющих {cumulative_variance[self.n_components_-1]:.3f} дисперсии")
        
        # Пересоздаем SVD с оптимальным количеством компонент
        self.svd_ = TruncatedSVD(n_components=self.n_components_)
        self.svd_.fit(X)
        
        return self
        
    def transform(self, X):
        return self.svd_.transform(X)

def make_plot_pipeline(variance_threshold=0.95, max_components=300):
    """
    Создает пайплайн для обработки текстового описания фильма.
    
    Args:
        variance_threshold (float): Минимальный порог объясненной дисперсии (0-1).
            По умолчанию 0.95 (95% информации).
        max_components (int): Максимальное количество компонент.
            По умолчанию 300.
    """
    return Pipeline([
        ('preprocess', TextPreprocessor()),
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('lsa', DynamicLSA(variance_threshold=variance_threshold, max_components=max_components))
    ])

def make_multi_label_pipeline():
    return Pipeline([
        ('split', SplitList()),
        ('binarizer', MultiLabelBinarizerTransformer())
    ])

def make_writers_pipeline():
    return Pipeline([
        ('split', SplitList()),
        ('binarize', MultiLabelBinarizerTransformer()),
        ('scale', StandardScaler(with_mean=False))  # Масштабирование без центрирования
    ])

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ('plot', make_plot_pipeline(variance_threshold=0.95, max_components=300), 'plot'),
            ('genres', make_multi_label_pipeline(), 'genres'),
            ('directors', make_multi_label_pipeline(), 'directors'),
            ('writers', make_writers_pipeline(), 'writers'),
            ('actors', make_multi_label_pipeline(), 'actors'),
            ('countries', make_multi_label_pipeline(), 'countries'),
            ('start_year', make_numeric_pipeline(), 'start_year'),
            ('type', Pipeline([
                ('preprocess', TextPreprocessor()),
                ('vectorizer', CountVectorizer(token_pattern='[^,]+'))
            ]), 'type'),
        ],
        sparse_threshold=1.0
    )
