import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const movieService = {
    // Получение списка фильмов с пагинацией
    getMovies: async (page = 1, pageSize = 20) => {
        try {
            const response = await axios.get(`${API_URL}/movies`, {
                params: { page, page_size: pageSize }
            });
            return response.data;
        } catch (error) {
            console.error('Error fetching movies:', error);
            throw error;
        }
    },

    // Получение информации о конкретном фильме
    getMovie: async (movieId) => {
        try {
            const response = await axios.get(`${API_URL}/movies/${movieId}`);
            return response.data;
        } catch (error) {
            console.error('Error fetching movie:', error);
            throw error;
        }
    }
}; 