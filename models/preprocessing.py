from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import issparse

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

def make_plot_pipeline(n_components=300):
    return Pipeline([
        ('preprocess', TextPreprocessor()),  # Добавляем предобработку текста
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('lsa', TruncatedSVD(n_components=min(n_components, 5)))  # Используем не более 5 компонент для тестов
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
            ('plot', make_plot_pipeline(n_components=300), 'plot'),
            ('genres', make_multi_label_pipeline(), 'genres'),
            ('directors', make_multi_label_pipeline(), 'directors'),
            ('writers', make_writers_pipeline(), 'writers'),
            ('actors', make_multi_label_pipeline(), 'actors'),
            ('countries', make_multi_label_pipeline(), 'countries'),
            ('start_year', make_numeric_pipeline(), 'start_year'),
            ('type', Pipeline([
                ('preprocess', TextPreprocessor()),  # Добавляем предобработку текста
                ('vectorizer', CountVectorizer(token_pattern='[^,]+'))
            ]), 'type'),
        ],
        sparse_threshold=1.0
    )
