from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

# Классы для строковых импутеров и токенизации
class SplitList(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [
            [item.strip() for item in (entry or '').split(',') if item and item.strip()]
            if entry else []  # Явно возвращаем пустой список для пропущенных значений
            for entry in X
        ]

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
        ('preprocess', TextPreprocessor()),
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('lsa', TruncatedSVD(n_components=n_components, random_state=9))
    ])

def make_multi_label_pipeline():
    return Pipeline([
        ('split', SplitList()),
        ('binarizer', MultiLabelBinarizerTransformer())
    ])

def make_writers_pipeline():
    return Pipeline([
        ('split', SplitList()),
        ('binarizer', MultiLabelBinarizerTransformer()),
        ('scale', FunctionTransformer(scale_half, accept_sparse=True))
    ])

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ('Plot ', make_plot_pipeline(n_components=300), 'Plot'),
            ('Genres', make_multi_label_pipeline(), 'Genres'),
            ('Directors', make_multi_label_pipeline(), 'Directors'),
            ('Writers', make_writers_pipeline(), 'Writers'),
            ('Actors', make_multi_label_pipeline(), 'Actors'),
            ('Countries', make_multi_label_pipeline(), 'Countries'),
            ('Year', make_numeric_pipeline(), 'Year'),
            ('Type', Pipeline([
                ('preprocess', TextPreprocessor()),
                ('vectorizer', CountVectorizer(token_pattern='[^,]+'))
            ]), 'Type'),
        ],
        sparse_threshold=1.0
    )