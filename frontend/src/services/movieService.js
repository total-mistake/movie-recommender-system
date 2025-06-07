const API_URL = 'http://localhost:8080/api';

export const SortField = {
    TITLE: 'title',
    YEAR: 'year',
    RATING: 'rating',
    RATING_COUNT: 'rating_count'
};

export const fetchMovies = async (page = 1, sortBy = SortField.RATING, sortOrder = 'desc') => {
    try {
        const response = await fetch(
            `${API_URL}/movies?page=${page}&sort_by=${sortBy}&sort_order=${sortOrder}`
        );
        if (!response.ok) {
            throw new Error('Failed to fetch movies');
        }
        return await response.json();
    } catch (error) {
        console.error('Error fetching movies:', error);
        throw error;
    }
};

export const fetchMovieById = async (movieId) => {
    try {
        const response = await fetch(`${API_URL}/movies/${movieId}`);
        if (!response.ok) {
            throw new Error('Failed to fetch movie');
        }
        return await response.json();
    } catch (error) {
        console.error('Error fetching movie:', error);
        throw error;
    }
};
