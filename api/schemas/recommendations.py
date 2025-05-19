from pydantic import BaseModel, Field
from typing import List, Optional

class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="ID of the user to get recommendations for")
    top_n: Optional[int] = Field(10, description="Number of recommendations to return")
    
class MovieIdRequest(BaseModel):
    movie_id: int = Field(..., description="ID of the movie to get similar movies for")
    top_n: Optional[int] = Field(10, description="Number of similar movies to return")

class RecommendationResponse(BaseModel):
    movie_ids: List[int] 