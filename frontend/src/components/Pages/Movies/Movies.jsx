import React, {useEffect, useState} from 'react';
import {useNavigate} from 'react-router-dom';
import Header from "../../Header/Header";
import Footer from "../../Footer/Footer";
import style from './Movies.module.css'
import Filter from "../../UI/Filter/Filter";
import { fetchMovies, SortField } from '../../../services/movieService';

const Movies = () => {
    const navigate = useNavigate();
    const [movies, setMovies] = useState([]);
    const [currentPage, setCurrentPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    const [isLoading, setIsLoading] = useState(false);
    const [hasMore, setHasMore] = useState(true);
    const [sortBy, setSortBy] = useState(SortField.PIPULARITY);
    const [sortOrder, setSortOrder] = useState('desc');
    const [searchQuery, setSearchQuery] = useState('');

    const handleMovieClick = (id) => {
        navigate(`/movie/${id}`);
    };

    const loadMovies = async (page, sortByField = sortBy, order = sortOrder) => {
        try {
            setIsLoading(true);
            const data = await fetchMovies(page, sortByField, order);
            
            if (page === 1) {
                setMovies(data.movies);
            } else {
                setMovies(prev => [...prev, ...data.movies]);
            }
            
            setTotalPages(data.total_pages);
            setHasMore(page < data.total_pages);
            setCurrentPage(page);
        } catch (error) {
            console.error('Error fetching movies:', error);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        setMovies([]);
        setCurrentPage(1);
        loadMovies(1, sortBy, sortOrder);
    }, [sortBy, sortOrder]);

    // Обработчик прокрутки для бесконечной подгрузки
    const handleScroll = (e) => {
        const { scrollTop, clientHeight, scrollHeight } = e.target.documentElement;
        if (scrollHeight - scrollTop <= clientHeight * 1.5 && !isLoading && hasMore) {
            loadMovies(currentPage + 1);
        }
    };

    useEffect(() => {
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, [isLoading, hasMore, currentPage, sortBy, sortOrder]);

    // Поиск фильмов по названию
    const handleSearch = (query) => {
        setSearchQuery(query);
        const filtered = movies.filter(movie =>
            movie.Title.toLowerCase().includes(query.toLowerCase())
        );
        setMovies(filtered);
    };

    // Сортировка фильмов
    const handleSort = (field) => {
        const newOrder = field === sortBy && sortOrder === 'desc' ? 'asc' : 'desc';
        setSortBy(field);
        setSortOrder(newOrder);
    };

    return (
        <body>
        <Header/>
        <div className={style.container14}>
            <div className={`${style.gallery3} thq-section-padding`}>
                <div className={`${style.maxWidth} thq-section-max-width`}>
                    <div className={style.sectionTitle}>
                        <Filter
                            onSearch={handleSearch}
                            onSort={handleSort}
                            currentSort={sortBy}
                            sortOrder={sortOrder}
                            sortOptions={[
                                { value: SortField.TITLE, label: 'По названию' },
                                { value: SortField.YEAR, label: 'По году' },
                                { value: SortField.RATING, label: 'По рейтингу' },
                                { value: SortField.RATING_COUNT, label: 'По количеству оценок' }
                            ]}
                        />
                    </div>
                    <div className={`${style.listOfMovies} thq-grid-4`}>
                        {movies.map(movie => (
                            <div key={movie.Movie_ID} 
                                 className={style.movie}
                                 onClick={() => handleMovieClick(movie.Movie_ID)}>
                                <img
                                    alt={movie.Title}
                                    src={movie.Poster}
                                    className={`${style.image} thq-img-ratio-16-9`}
                                />
                                <div className={style.movieInfo}>
                                    <div className={style.movieTitle}>{movie.Title}</div>
                                    <div className={style.movieYear}>{movie.Year}</div>
                                    <div className={style.movieRating}>
                                        Рейтинг: {movie.Rating.toFixed(1)} ({movie.Rating_Count})
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                    {isLoading && <div className={style.loading}>Загрузка...</div>}
                </div>
            </div>
        </div>
        <Footer/>
        </body>
    );
};

export default Movies;