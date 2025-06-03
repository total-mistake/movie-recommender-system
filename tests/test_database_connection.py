import pytest
from database.connection import (
    get_movies_data,
    get_ratings_data,
    get_movie_by_id,
    add_movie_to_db,
    update_movie_in_db,
    delete_movie_from_db,
    add_rating_to_db,
    add_user_to_db,
    verify_user_credentials,
    get_genres,
    add_movie_by_imdb_id,
    get_complete_movie_details,
    get_user_ratings
)
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
import pandas as pd
import random
import string
from datetime import datetime

# Фикстуры для тестовых данных
@pytest.fixture
def sample_movie_data():
    return {
        'Title': 'Test Movie',
        'Type': 'movie',
        'IMDb_ID': 'tt1234567',
        'Plot': 'Test plot',
        'Year': 2024,
        'Poster': 'http://example.com/poster.jpg',
        'Genres': 'Action, Adventure',
        'Directors': 'Test Director',
        'Writers': 'Test Writer',
        'Actors': 'Test Actor',
        'Countries': 'USA'
    }

@pytest.fixture
def sample_movie_update_data():
    return {
        'Movie_ID': 292772,
        'Title': 'Updated Test Movie',
        'Type': 'movie',
        'IMDb_ID': 'tt1234567',
        'Plot': 'Updated test plot',
        'Year': 2024,
        'Poster': 'http://example.com/updated_poster.jpg',
        'Genres': 'Action, Adventure, Drama',
        'Directors': 'Updated Test Director',
        'Writers': 'Updated Test Writer',
        'Actors': 'Updated Test Actor',
        'Countries': 'USA, UK'
    }

@pytest.fixture
def random_username():
    """Генерирует случайное имя пользователя для тестов"""
    return ''.join(random.choices(string.ascii_lowercase, k=8))

# Тесты для получения данных
def test_get_movies_data():
    """Тест получения данных о фильмах"""
    movies = get_movies_data()
    assert isinstance(movies, pd.DataFrame)
    assert not movies.empty
    assert 'Movie_ID' in movies.columns
    assert 'Title' in movies.columns
    assert 'Rating' in movies.columns
    assert 'Rating_Count' in movies.columns

def test_get_ratings_data():
    """Тест получения данных о рейтингах"""
    ratings = get_ratings_data()
    assert isinstance(ratings, pd.DataFrame)
    assert not ratings.empty
    assert 'User_ID' in ratings.columns
    assert 'Movie_ID' in ratings.columns
    assert 'Rating' in ratings.columns

def test_get_movie_by_id():
    """Тест получения информации о конкретном фильме"""
    movie = get_movie_by_id(1)
    assert isinstance(movie, dict)
    assert 'Movie_ID' in movie
    assert 'Title' in movie
    assert 'Rating' in movie
    assert 'Rating_Count' in movie
    assert movie['Movie_ID'] == 1
    assert isinstance(movie['Rating'], (float, type(None)))
    assert isinstance(movie['Rating_Count'], (int, type(None)))

def test_get_movie_by_id_nonexistent():
    """Тест получения информации о несуществующем фильме"""
    movie = get_movie_by_id(999999)  # Предполагаем, что такого ID нет
    assert movie is None

# Тесты для работы с фильмами
def test_add_movie_to_db(sample_movie_data):
    """Тест добавления нового фильма"""
    try:
        add_movie_to_db(sample_movie_data)
        # Проверяем, что фильм добавился
        movies = get_movies_data()
        assert any(movies['Title'] == sample_movie_data['Title'])
    except SQLAlchemyError as e:
        pytest.fail(f"Ошибка при добавлении фильма: {str(e)}")

def test_update_movie_in_db(sample_movie_update_data):
    """Тест обновления информации о фильме"""
    try:
        update_movie_in_db(sample_movie_update_data)
        # Проверяем, что фильм обновился
        movie = get_movie_by_id(sample_movie_update_data['Movie_ID'])
        assert movie['Title'] == sample_movie_update_data['Title']
    except SQLAlchemyError as e:
        pytest.fail(f"Ошибка при обновлении фильма: {str(e)}")

def test_delete_movie_from_db():
    """Тест удаления фильма"""
    try:
        # Сначала добавляем тестовый фильм
        test_movie = {
            'Title': 'Movie to Delete',
            'Type': 'movie',
            'IMDb_ID': 'tt7654321',
            'Plot': 'Test plot for deletion',
            'Year': 2024,
            'Poster': 'http://example.com/delete.jpg',
            'Genres': 'Action',
            'Directors': 'Test Director',
            'Writers': 'Test Writer',
            'Actors': 'Test Actor',
            'Countries': 'USA'
        }
        add_movie_to_db(test_movie)
        
        # Получаем ID добавленного фильма
        movies = get_movies_data()
        movie_id = movies[movies['Title'] == test_movie['Title']]['Movie_ID'].iloc[0]
        
        # Удаляем фильм
        delete_movie_from_db(movie_id)
        
        # Проверяем, что фильм удален
        movie = get_movie_by_id(movie_id)
        assert movie is None
    except SQLAlchemyError as e:
        pytest.fail(f"Ошибка при удалении фильма: {str(e)}")

# Тесты для работы с рейтингами
def test_add_rating_to_db():
    """Тест добавления рейтинга"""
    try:
        user_id = 1
        movie_id = 1
        rating = 4.5
        add_rating_to_db(user_id, movie_id, rating)
        
        # Проверяем, что рейтинг добавился через get_user_ratings
        ratings = get_user_ratings(user_id)
        assert len(ratings) > 0
        
        # Ищем добавленный рейтинг
        user_rating = next(
            (r for r in ratings if r['Movie_ID'] == movie_id and r['Rating'] == rating),
            None
        )
        assert user_rating is not None
        assert user_rating['Rating'] == rating
        assert 'Title' in user_rating
        assert 'Date_Rated' in user_rating
    except SQLAlchemyError as e:
        pytest.fail(f"Ошибка при добавлении рейтинга: {str(e)}")

# Тесты для работы с пользователями
def test_add_user_to_db(random_username):
    """Тест добавления нового пользователя"""
    try:
        password = "testpass123"
        user_id = add_user_to_db(random_username, password)
        assert isinstance(user_id, int)
        assert user_id > 0
    except SQLAlchemyError as e:
        pytest.fail(f"Ошибка при добавлении пользователя: {str(e)}")

def test_verify_user_credentials(random_username):
    """Тест проверки учетных данных пользователя"""
    try:
        password = "testpass123"
        user_id = add_user_to_db(random_username, password)
        verified_id = verify_user_credentials(random_username, password)
        assert verified_id == user_id
        wrong_id = verify_user_credentials(random_username, "wrongpass")
        assert wrong_id is None
    except SQLAlchemyError as e:
        pytest.fail(f"Ошибка при проверке учетных данных: {str(e)}")

# Тесты для работы с жанрами
def test_get_genres():
    """Тест получения списка жанров"""
    genres = get_genres()
    assert isinstance(genres, list)
    assert len(genres) > 0
    assert all(isinstance(genre, dict) for genre in genres)
    assert all('Genre_ID' in genre and 'Genre' in genre for genre in genres)

# Тесты для работы с IMDb API
@pytest.mark.parametrize("imdb_id,expected_title", [
    ("tt29344903", "The Ugly Stepsister"),  # Действительный ID
    ("tt9999999", None),  # Несуществующий ID
    ("invalid_id", None),  # Неверный формат ID
])
def test_add_movie_by_imdb_id(imdb_id, expected_title):
    """Параметризованный тест добавления фильма по IMDb ID"""
    try:
        movie_data = add_movie_by_imdb_id(imdb_id)
        if expected_title is None:
            assert movie_data is None
        else:
            assert movie_data is not None
            assert isinstance(movie_data, dict)
            assert 'Title' in movie_data
            assert movie_data['Title'] == expected_title
            if 'Movie_ID' in movie_data:
                delete_movie_from_db(movie_data['Movie_ID'])
    except Exception as e:
        if expected_title is None:
            assert movie_data is None
        else:
            raise e

def test_add_movie_by_imdb_id_compare_with_api():
    """Тест сравнения данных фильма из БД и API"""
    imdb_id = "tt29344903"  # The Ugly Stepsister
    
    # Получаем данные из API
    api_data = get_complete_movie_details(imdb_id)
    if api_data is None or 'data' not in api_data or 'title' not in api_data['data']:
        pytest.skip("API недоступен или вернул неверные данные")
    
    # Добавляем фильм в БД
    db_movie_data = add_movie_by_imdb_id(imdb_id)
    db_movie_data = get_movie_by_id(db_movie_data['Movie_ID'])
    assert db_movie_data is not None
    
    # Сравниваем данные
    api_title = api_data['data']['title']
    assert db_movie_data['Title'] == api_title['primary_title']
    assert db_movie_data['Type'] == api_title['type']
    assert db_movie_data['Plot'] == api_title['plot']
    assert db_movie_data['Year'] == api_title.get('start_year')
    assert db_movie_data['Genres'] == ', '.join(api_title.get('genres') or [])
    delete_movie_from_db(db_movie_data['Movie_ID'])

# Тесты на обработку ошибок
def test_add_movie_invalid_data():
    """Тест добавления фильма с неверными данными"""
    invalid_data = {
        'Type': 'movie',
        # Отсутствуют обязательные поля
    }
    with pytest.raises((SQLAlchemyError, ValueError)):
        add_movie_to_db(invalid_data)

def test_add_rating_invalid_data():
    """Тест добавления рейтинга с неверными данными"""
    with pytest.raises(SQLAlchemyError):
        add_rating_to_db(user_id=-1, movie_id=-1, rating=6.0)  # Неверный рейтинг

def test_verify_user_credentials_nonexistent():
    """Тест проверки учетных данных несуществующего пользователя"""
    result = verify_user_credentials("nonexistent_user", "password")
    assert result is None

def test_get_user_ratings():
    """Тест получения списка рейтингов пользователя"""
    user_id = 1
    ratings = get_user_ratings(user_id)
    assert isinstance(ratings, list)
    assert len(ratings) > 0
    for rating in ratings:
        assert 'Movie_ID' in rating
        assert 'Rating' in rating
        assert 'Title' in rating
        assert 'Date_Rated' in rating
        assert isinstance(rating['Movie_ID'], int)
        assert isinstance(rating['Rating'], (int, float))
        assert isinstance(rating['Title'], str)
        assert isinstance(rating['Date_Rated'], (str, datetime))
    # Проверяем сортировку по дате (от новых к старым)
    if len(ratings) > 1:
        assert ratings[0]['Date_Rated'] >= ratings[1]['Date_Rated'] 