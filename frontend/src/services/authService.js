const API_URL = '/api/auth';

export const authService = {
    // Регистрация нового пользователя
    async register(userData) {
        try {
            const response = await fetch(`${API_URL}/register`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                credentials: 'include',
                body: JSON.stringify(userData)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Registration failed');
            }

            return await response.json();
        } catch (error) {
            throw error;
        }
    },

    // Вход пользователя
    async login(credentials) {
        try {
            const response = await fetch(`${API_URL}/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                credentials: 'include',
                body: JSON.stringify(credentials)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Login failed');
            }

            const data = await response.json();
            // Сохраняем данные пользователя в localStorage
            localStorage.setItem('user', JSON.stringify(data));
            return data;
        } catch (error) {
            throw error;
        }
    },

    // Выход пользователя
    logout() {
        localStorage.removeItem('user');
    },

    // Получение текущего пользователя
    getCurrentUser() {
        const user = localStorage.getItem('user');
        return user ? JSON.parse(user) : null;
    }
}; 