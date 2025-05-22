import unittest
import numpy as np
import pandas as pd
import os
import sys
from scipy.sparse import issparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.preprocessing import (
    SplitList,
    MultiLabelBinarizerTransformer,
    make_numeric_pipeline,
    make_plot_pipeline,
    make_multi_label_pipeline,
    make_writers_pipeline,
    build_preprocessor
)

class TestPreprocessing(unittest.TestCase):
    """
    Тесты для модуля предобработки данных.
    
    Проверяет корректность работы функций предобработки текстовых данных
    и создания признаков для фильмов.
    """
    
    def setUp(self):
        # Создание тестовых данных
        self.test_data = pd.DataFrame({
            'genres': ['Action,Adventure', 'Comedy,Romance', 'Drama,Thriller', 'Action,Comedy'],
            'directors': ['Director1,Director2', 'Director3', 'Director4', 'Director5'],
            'writers': ['Writer1,Writer2', 'Writer3', 'Writer4', 'Writer5'],
            'actors': ['Actor1,Actor2,Actor3', 'Actor4,Actor5', 'Actor6,Actor7', 'Actor8,Actor9'],
            'countries': ['USA,UK', 'France', 'Germany', 'Italy'],
            'plot': ['Action movie plot with exciting scenes and car chases', 
                    'Comedy movie plot with funny moments and jokes',
                    'Drama movie plot with emotional scenes and character development',
                    'Thriller movie plot with suspense and plot twists'],
            'start_year': [2000, 2010, 2020, 2015],
            'type': ['movie', 'movie', 'movie', 'movie']
        })

    def test_split_list(self):
        """Тест разделения строки с разделителями на список элементов"""
        splitter = SplitList()
        test_input = ['a,b,c', 'd,e']
        expected = [['a', 'b', 'c'], ['d', 'e']]
        result = splitter.transform(test_input)
        self.assertEqual(result, expected)

    def test_split_list_with_empty_and_none(self):
        """Проверка корректности SplitList при пустых строках и None"""
        splitter = SplitList()
        test_input = ['a,b', '', None]
        result = splitter.transform(test_input)
        expected = [['a', 'b'], [], []]
        self.assertEqual(result, expected)

    def test_multi_label_binarizer(self):
        binarizer = MultiLabelBinarizerTransformer()
        test_input = [['a', 'b'], ['b', 'c']]
        binarizer.fit(test_input)
        result = binarizer.transform(test_input).toarray()
        expected = np.array([[1, 1, 0], [0, 1, 1]])  # порядок зависит от fit
        np.testing.assert_array_equal(result, expected)

    def test_multi_label_binarizer_with_unseen_labels(self):
        """Проверка обработки новых меток после обучения"""
        binarizer = MultiLabelBinarizerTransformer()
        binarizer.fit([['a', 'b']])
        transformed = binarizer.transform([['b', 'c']])
        # 'c' не встречался при обучении, он будет игнорироваться
        self.assertEqual(transformed.shape[1], 2)

    def test_numeric_pipeline(self):
        """Тест обработки числовых данных (масштабирование и импутация)"""
        pipeline = make_numeric_pipeline()
        test_data = self.test_data[['start_year']]
        result = pipeline.fit_transform(test_data)
        self.assertEqual(result.shape[0], 4)
        self.assertEqual(result.shape[1], 1)

    def test_numeric_pipeline_with_nan(self):
        """Проверка обработки пропущенных значений в числовом пайплайне"""
        pipeline = make_numeric_pipeline()
        test_data = pd.DataFrame({'start_year': [2000, np.nan, 2010, 2020]})
        result = pipeline.fit_transform(test_data)
        self.assertEqual(result.shape, (4, 1))

    def test_plot_pipeline(self):
        """Тест обработки текстовых данных сюжета (TF-IDF и LSA)"""
        pipeline = make_plot_pipeline(n_components=2)
        test_data = self.test_data['plot']
        result = pipeline.fit_transform(test_data)
        self.assertEqual(result.shape[0], 4)
        self.assertEqual(result.shape[1], 2)

    def test_plot_pipeline_with_empty_strings(self):
        """Проверка работы текстового пайплайна с пустыми строками"""
        pipeline = make_plot_pipeline(n_components=1)  # заменили 2 на 1
        test_data = pd.Series(['', 'word1', ' ', 'word2'])
        result = pipeline.fit_transform(test_data)
        self.assertEqual(result.shape[0], 4)

    def test_multi_label_pipeline(self):
        """Тест обработки мультиметочных данных (жанры, актеры и т.д.)"""
        pipeline = make_multi_label_pipeline()
        test_data = self.test_data['genres']
        result = pipeline.fit_transform(test_data)
        self.assertTrue(issparse(result) or isinstance(result, np.ndarray))
        self.assertEqual(result.shape[0], 4)

    def test_writers_pipeline(self):
        """Тест обработки данных о сценаристах с дополнительным масштабированием"""
        pipeline = make_writers_pipeline()
        test_data = self.test_data['writers']
        result = pipeline.fit_transform(test_data)
        self.assertTrue(issparse(result) or isinstance(result, np.ndarray))
        self.assertEqual(result.shape[0], 4)

    def test_full_preprocessor(self):
        """Тест полного препроцессора, объединяющего все пайплайны"""
        preprocessor = build_preprocessor()
        result = preprocessor.fit_transform(self.test_data)
        self.assertTrue(issparse(result) or isinstance(result, np.ndarray))
        self.assertEqual(result.shape[0], 4)

if __name__ == '__main__':
    unittest.main() 