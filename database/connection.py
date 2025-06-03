from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
import pandas as pd
import requests
import hashlib
from typing import Generator, Dict, Any, List, Optional, Tuple
from .models import MovieView, Movie, Rating
from config import (
    DATABASE_URL, IMDB_API_URL, IMDB_API_TIMEOUT
)

# Создаем движок базы данных
engine = create_engine(DATABASE_URL)

# Создаем фабрику сессий
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Контекстный менеджер для работы с сессиями базы данных.
    Автоматически закрывает сессию после использования.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_movies_data() -> pd.DataFrame:
    """
    Получение данных о фильмах из базы данных.
    """
    return pd.read_sql("SELECT * FROM movieview", engine)

def get_ratings_data() -> pd.DataFrame:
    """
    Получение данных о рейтингах из базы данных.
    """
    return pd.read_sql("SELECT * FROM rating", engine)

def get_movie_by_id(movie_id: int) -> Dict[str, Any]:
    """
    Получение информации о конкретном фильме по его ID.
    Использует представление movieview для получения данных.
    """
    with get_db() as db:
        movie = db.query(MovieView).filter(MovieView.Movie_ID == movie_id).first()
        if movie is None:
            return None
        return {
            'Movie_ID': movie.Movie_ID,
            'Title': movie.Title,
            'Type': movie.Type,
            'Plot': movie.Plot,
            'Year': movie.Year,
            'Poster': movie.Poster,
            'Genres': movie.Genres,
            'Directors': movie.Directors,
            'Writers': movie.Writers,
            'Actors': movie.Actors,
            'Countries': movie.Countries,
            'Rating': movie.Rating,
            'Rating_Count': movie.Rating_Count
        }

def add_movie_to_db(movie_data: Dict[str, Any]) -> int:
    """
    Добавление нового фильма в базу данных через хранимую процедуру.
    Возвращает ID созданного фильма.
    """
    with get_db() as db:
        query = text("""
            CALL AddMovie(
                :Title, :Type, :IMDb_ID, :Plot, :Year,
                :Poster, :Genres, :Directors, :Writers,
                :Actors, :Countries, @Movie_ID
            )
        """)
        db.execute(query, {
            'Title': movie_data.get('Title'),
            'Type': movie_data.get('Type'),
            'IMDb_ID': movie_data.get('IMDb_ID'),
            'Plot': movie_data.get('Plot'),
            'Year': movie_data.get('Year'),
            'Poster': movie_data.get('Poster'),
            'Genres': movie_data.get('Genres'),
            'Directors': movie_data.get('Directors'),
            'Writers': movie_data.get('Writers'),
            'Actors': movie_data.get('Actors'),
            'Countries': movie_data.get('Countries')
        })
        
        # Получаем ID созданного фильма
        result = db.execute(text("SELECT @Movie_ID"))
        movie_id = result.scalar()
        db.commit()
        return movie_id

def make_graphql_request(query: str, variables: Optional[Dict] = None) -> Optional[Dict]:
    """
    Выполнение GraphQL-запросов к IMDb API
    """
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "query": query,
        "variables": variables if variables else {}
    }
    
    try:
        response = requests.post(
            IMDB_API_URL, 
            json=payload, 
            headers=headers, 
            timeout=IMDB_API_TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Ошибка: {response.status_code} — {response.text}")
            return None
    except Exception as e:
        print(f"Ошибка при выполнении запроса: {e}")
        return None

def get_complete_movie_details(movie_id: str) -> Optional[Dict]:
    """
    Получение полной информации о фильме по его IMDb ID
    """
    query = """
    query GetMovieDetails($id: ID!) {
        title(id: $id) {
            id
            type
            primary_title
            plot
            genres
            start_year
            directors: credits(first: 5, categories: ["director"]) {
                name {
                    display_name
                }
            }
            writers: credits(first: 5, categories: ["writer"]) {
                name {
                    display_name
                }
            }
            casts: credits(first: 5, categories: ["actor", "actress"]) {
                name {
                    display_name
                }
            }
            origin_countries {
                name
            }
            posters {
                url
            }
        }
    }
    """
    variables = {"id": movie_id}
    return make_graphql_request(query, variables)

def add_movie_by_imdb_id(imdb_id: str) -> Optional[Dict[str, Any]]:
    """
    Добавление фильма в базу данных по его IMDb ID.
    Возвращает данные добавленного фильма или None в случае ошибки.
    """
    # Получаем данные о фильме из IMDb API
    movie_data = get_complete_movie_details(imdb_id)
    if not movie_data or 'data' not in movie_data or 'title' not in movie_data['data']:
        return None
    
    title_data = movie_data['data']['title']
    if not title_data:
        return None
    
    try:
        # Формируем данные для добавления в БД
        db_movie_data = {
            'Title': title_data.get('primary_title', ''),
            'Type': title_data.get('type', ''),
            'IMDb_ID': imdb_id,
            'Plot': title_data.get('plot', ''),
            'Year': title_data.get('start_year', ''),
            'Poster': title_data.get('posters', [{}])[0].get('url', '') if title_data.get('posters') else '',
            'Genres': ', '.join(title_data.get('genres') or []),
            'Directors': ', '.join(d.get('name', {}).get('display_name', '') for d in title_data.get('directors') or []),
            'Writers': ', '.join(w.get('name', {}).get('display_name', '') for w in title_data.get('writers') or []),
            'Actors': ', '.join(a.get('name', {}).get('display_name', '') for a in title_data.get('casts') or []),
            'Countries': ', '.join(c.get('name', '') for c in title_data.get('origin_countries') or [])
        }
        
        # Добавляем фильм в БД
        movie_id = add_movie_to_db(db_movie_data)
        db_movie_data['Movie_ID'] = movie_id
        return db_movie_data
    except Exception as e:
        print(f"Ошибка при добавлении фильма в БД: {e}")
        return None

def get_genres() -> List[Dict[str, Any]]:
    """
    Получение списка всех жанров из базы данных.
    Возвращает список словарей с id и названием жанра.
    """
    with get_db() as db:
        query = text("SELECT Genre_ID, Genre FROM genre ORDER BY Genre")
        result = db.execute(query)
        return [{"Genre_ID": row[0], "Genre": row[1]} for row in result]

def update_movie_in_db(movie_data: Dict[str, Any]) -> None:
    """
    Обновление информации о фильме в базе данных через хранимую процедуру.
    """
    with get_db() as db:
        query = text("""
            CALL UpdateMovie(
                :Movie_ID, :Title, :Type, :IMDb_ID, :Plot, :Year,
                :Poster, :Genres, :Directors, :Writers,
                :Actors, :Countries
            )
        """)
        db.execute(query, {
            'Movie_ID': movie_data['Movie_ID'],
            'Title': movie_data.get('Title'),
            'Type': movie_data.get('Type'),
            'IMDb_ID': movie_data.get('IMDb_ID'),
            'Plot': movie_data.get('Plot'),
            'Year': movie_data.get('Year'),
            'Poster': movie_data.get('Poster'),
            'Genres': movie_data.get('Genres'),
            'Directors': movie_data.get('Directors'),
            'Writers': movie_data.get('Writers'),
            'Actors': movie_data.get('Actors'),
            'Countries': movie_data.get('Countries')
        })
        db.commit()

def delete_movie_from_db(movie_id: int) -> None:
    """
    Удаление фильма из базы данных через хранимую процедуру.
    """
    with get_db() as db:
        query = text("CALL DeleteMovie(:Movie_ID)")
        db.execute(query, {'Movie_ID': movie_id})
        db.commit()

def add_rating_to_db(user_id: int, movie_id: int, rating: float) -> None:
    """
    Добавление рейтинга пользователя через хранимую процедуру.
    """
    with get_db() as db:
        query = text("CALL AddRating(:User_ID, :Movie_ID, :Rating)")
        db.execute(query, {
            'User_ID': user_id,
            'Movie_ID': movie_id,
            'Rating': rating
        })
        db.commit()

def hash_password(password: str) -> str:
    """
    Хеширование пароля с использованием SHA-256
    """
    return hashlib.sha256(password.encode()).hexdigest()

def add_user_to_db(username: str, password: str) -> int:
    """
    Добавление нового пользователя в базу данных.
    Возвращает ID созданного пользователя.
    """
    password_hash = hash_password(password)
    with get_db() as db:
        query = text("""
            CALL AddUser(:Username, :PasswordHash, @User_ID)
        """)
        db.execute(query, {
            'Username': username,
            'PasswordHash': password_hash
        })
        
        # Получаем ID созданного пользователя
        result = db.execute(text("SELECT @User_ID"))
        user_id = result.scalar()
        db.commit()
        return user_id

def verify_user_credentials(username: str, password: str) -> Optional[int]:
    """
    Проверка учетных данных пользователя.
    Возвращает ID пользователя при успешной аутентификации или None.
    """
    password_hash = hash_password(password)
    with get_db() as db:
        query = text("""
            SELECT User_ID 
            FROM user 
            WHERE Login = :Login AND Password = :Password
        """)
        result = db.execute(query, {
            'Login': username,
            'Password': password_hash
        }).scalar()
        return result

def get_user_ratings(user_id: int) -> List[Dict[str, Any]]:
    """
    Получение всех рейтингов пользователя.
    Возвращает список словарей с информацией о рейтингах.
    """
    with get_db() as db:
        query = text("""
            SELECT r.Movie_ID, r.Rating, r.Date_Rated, m.Title
            FROM rating r
            JOIN movie m ON r.Movie_ID = m.Movie_ID
            WHERE r.User_ID = :User_ID
            ORDER BY r.Date_Rated DESC
        """)
        result = db.execute(query, {'User_ID': user_id})
        return [
            {
                'Movie_ID': row[0],
                'Rating': row[1],
                'Date_Rated': row[2],
                'Title': row[3]
            }
            for row in result
        ]