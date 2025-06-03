import React from 'react';
import { Routes, Route } from 'react-router-dom';
import HomePage from './components/Pages/HomePage/HomePage';
import Movies from './components/Pages/Movies/Movies';
import Movie from './components/Pages/Movie/Movie';
import Login from './components/Pages/Login/Login';

const App = () => {
    return (
        <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/movies" element={<Movies />} />
            <Route path="/movie/:id" element={<Movie />} />
            <Route path="/login" element={<Login />} />
        </Routes>
    );
};

export default App; 