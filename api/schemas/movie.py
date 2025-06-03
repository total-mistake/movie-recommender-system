from pydantic import BaseModel
from typing import List, Optional

class MovieBase(BaseModel):
    Movie_ID: int
    Title: str
    Type: Optional[str] = None
    Plot: Optional[str] = None
    Year: Optional[int] = None
    Poster: Optional[str] = None
    Genres: Optional[str] = None
    Directors: Optional[str] = None
    Writers: Optional[str] = None
    Actors: Optional[str] = None
    Countries: Optional[str] = None
    Rating: Optional[float] = None
    Rating_Count: Optional[int] = None

class MovieResponse(MovieBase):
    class Config:
        from_attributes = True

class MovieListResponse(BaseModel):
    movies: List[MovieResponse]
    total: int
    page: int
    page_size: int
    total_pages: int 