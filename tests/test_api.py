import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.auth import create_access_token
from unittest.mock import patch, MagicMock
import json

@pytest.fixture(autouse=True)
def setup_app():
    """Фикстура для инициализации состояния приложения перед каждым тестом"""
    app.state.is_ready = True
    yield
    app.state.is_ready = False

client = TestClient(app)

# Test data
TEST_USER = {
    "username": "testuser",
    "password": "testpass",
    "user_id": 1
}

TEST_MOVIE = {
    "id": 1,
    "title": "Test Movie",
    "genres": ["Action", "Drama"],
    "rating": 4.5
}

@pytest.fixture
def auth_headers():
    """Fixture to create authentication headers"""
    token = create_access_token(TEST_USER["user_id"])
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def mock_recommender_service():
    """Fixture to mock recommender service"""
    with patch("api.dependencies.get_recommender_service") as mock:
        service = MagicMock()
        service.get_recommendations.return_value = [1, 2, 3]
        service.get_recent_recommendations.return_value = [4, 5, 6]
        service.get_similar_movies.return_value = [7, 8, 9]
        mock.return_value = service
        yield service

@pytest.fixture
def mock_db():
    """Fixture to mock database functions"""
    with patch("database.connection.get_movies_data") as mock_movies, \
         patch("database.connection.get_movie_by_id") as mock_movie, \
         patch("database.connection.get_genres") as mock_genres, \
         patch("database.connection.add_user_to_db") as mock_add_user, \
         patch("database.connection.verify_user_credentials") as mock_verify:
        
        mock_movies.return_value = [TEST_MOVIE]
        mock_movie.return_value = TEST_MOVIE
        mock_genres.return_value = ["Action", "Drama", "Comedy"]
        mock_add_user.return_value = TEST_USER["user_id"]
        mock_verify.return_value = TEST_USER["user_id"]
        
        yield {
            "movies": mock_movies,
            "movie": mock_movie,
            "genres": mock_genres,
            "add_user": mock_add_user,
            "verify": mock_verify
        }

# Auth tests
def test_register_user(mock_db):
    """Test user registration endpoint"""
    response = client.post(
        "/register",
        json={"username": TEST_USER["username"], "password": TEST_USER["password"]}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["user_id"] == TEST_USER["user_id"]
    assert data["token_type"] == "bearer"

def test_login_user(mock_db):
    """Test user login endpoint"""
    response = client.post(
        "/login",
        json={"username": TEST_USER["username"], "password": TEST_USER["password"]}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["user_id"] == TEST_USER["user_id"]

def test_get_current_user(auth_headers):
    """Test getting current user info"""
    response = client.get("/me", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["user_id"] == TEST_USER["user_id"]

# Movie tests
def test_get_movies(mock_db, auth_headers):
    """Test getting movies list"""
    response = client.get("/movies/", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "movies" in data
    assert len(data["movies"]) > 0
    assert "total" in data
    assert "page" in data
    assert "page_size" in data

def test_get_movie(mock_db, auth_headers):
    """Test getting single movie"""
    response = client.get(f"/movies/{TEST_MOVIE['id']}", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == TEST_MOVIE["id"]
    assert data["title"] == TEST_MOVIE["title"]

def test_get_similar_movies(mock_recommender_service, auth_headers):
    """Test getting similar movies"""
    response = client.post(
        "/movies/similar",
        json={"movie_id": TEST_MOVIE["id"], "top_n": 3},
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert "movie_ids" in data
    assert len(data["movie_ids"]) == 3
    mock_recommender_service.get_similar_movies.assert_called_once_with(
        movie_id=TEST_MOVIE["id"],
        top_n=3
    )

def test_get_genres(mock_db, auth_headers):
    """Test getting genres list"""
    response = client.get("/movies/genres", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "Action" in data
    assert "Drama" in data

# User recommendations tests
def test_get_user_recommendations(mock_recommender_service, auth_headers):
    """Test getting user recommendations"""
    response = client.post(
        "/recommendations",
        json={"user_id": TEST_USER["user_id"], "top_n": 3},
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert "movie_ids" in data
    assert len(data["movie_ids"]) == 3
    mock_recommender_service.get_recommendations.assert_called_once_with(
        user_id=TEST_USER["user_id"],
        top_n=3
    )

def test_get_user_recent_recommendations(mock_recommender_service, auth_headers):
    """Test getting user recent recommendations"""
    response = client.post(
        "/recommendations/recent",
        json={"user_id": TEST_USER["user_id"], "top_n": 3},
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert "movie_ids" in data
    assert len(data["movie_ids"]) == 3
    mock_recommender_service.get_recent_recommendations.assert_called_once_with(
        user_id=TEST_USER["user_id"],
        top_n=3
    )

# Error cases
def test_unauthorized_access():
    """Test unauthorized access to protected endpoints"""
    endpoints = [
        ("/me", "GET"),
        ("/movies/", "GET"),
        ("/movies/1", "GET"),
        ("/movies/similar", "POST"),
        ("/recommendations", "POST"),
        ("/recommendations/recent", "POST")
    ]
    
    for endpoint, method in endpoints:
        if method == "GET":
            response = client.get(endpoint)
        else:
            response = client.post(endpoint, json={})
        assert response.status_code == 401

def test_invalid_movie_id(mock_db, auth_headers):
    """Test getting non-existent movie"""
    mock_db["movie"].return_value = None
    response = client.get("/movies/999", headers=auth_headers)
    assert response.status_code == 404

def test_invalid_recommendation_request(mock_recommender_service, auth_headers):
    """Test invalid recommendation request"""
    response = client.post(
        "/recommendations",
        json={"user_id": 999, "top_n": 3},  # Different user_id than authenticated
        headers=auth_headers
    )
    assert response.status_code == 403 