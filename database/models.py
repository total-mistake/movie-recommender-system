from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey, DateTime, Table
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class MovieView(Base):
    """
    Модель для представления movieview.
    Это представление, а не таблица, поэтому не имеет прямых связей.
    """
    __tablename__ = 'movieview'
    
    Movie_ID = Column(Integer, primary_key=True)
    Title = Column(String(255), nullable=False)
    Type = Column(String(50))
    Plot = Column(Text)
    Year = Column(Integer)
    Poster = Column(String(255))
    Genres = Column(String(255))
    Directors = Column(String(255))
    Writers = Column(String(255))
    Actors = Column(String(255))
    Countries = Column(String(255))
    Rating = Column(Float)  # Средний рейтинг фильма
    Rating_Count = Column(Integer)  # Количество отзывов

class Movie(Base):
    """
    Модель для таблицы movie.
    """
    __tablename__ = 'movie'
    
    Movie_ID = Column(Integer, primary_key=True, autoincrement=True)
    Title = Column(String(255), nullable=False)
    IMDb_ID = Column(String(50))
    Type = Column(String(50))
    Plot = Column(Text)
    Year = Column(Integer)
    Poster = Column(Text)
    
    # Связь с рейтингами
    ratings = relationship("Rating", back_populates="movie")

class Rating(Base):
    """
    Модель для таблицы rating.
    """
    __tablename__ = 'rating'
    
    Movie_ID = Column(Integer, ForeignKey('movie.Movie_ID'), primary_key=True)
    User_ID = Column(Integer, ForeignKey('user.User_ID'), primary_key=True)
    Rating = Column(Float)
    Date_Rated = Column(DateTime, default=datetime.utcnow)
    
    # Связь с фильмом
    movie = relationship("Movie", back_populates="ratings") 