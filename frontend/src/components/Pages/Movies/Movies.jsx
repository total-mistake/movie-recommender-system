import React, {useEffect, useState} from 'react';
import {useNavigate} from 'react-router-dom';
import Header from "../../Header/Header";
import Footer from "../../Footer/Footer";
import style from './Movies.module.css'
import Filter from "../../UI/Filter/Filter";
import { movieService } from '../../../../services/movieService';

const Movies = () => {
    const navigate = useNavigate();
    const [movies, setMovies] = useState([]);
    const [filteredMovies, setFilteredMovies] = useState([]);
    const [currentPage, setCurrentPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    const [isLoading, setIsLoading] = useState(false);
    const [hasMore, setHasMore] = useState(true);
    const pageSize = 20;

    const handleMovieClick = (id) => {
        console.log('Clicked movie id:', id);
        navigate(`/movie/${id}`);
    };

    const fetchMovies = async (page) => {
        try {
            setIsLoading(true);
            const data = await movieService.getMovies(page, pageSize);
            if (page === 1) {
                setMovies(data.movies);
                setFilteredMovies(data.movies);
            } else {
                setMovies(prev => [...prev, ...data.movies]);
                setFilteredMovies(prev => [...prev, ...data.movies]);
            }
            setTotalPages(data.total_pages);
            setHasMore(page < data.total_pages);
        } catch (error) {
            console.error('Error fetching movies:', error);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchMovies(1);
    }, []);

    // Обработчик прокрутки для бесконечной подгрузки
    const handleScroll = (e) => {
        const { scrollTop, clientHeight, scrollHeight } = e.target.documentElement;
        if (scrollHeight - scrollTop <= clientHeight * 1.5 && !isLoading && hasMore) {
            setCurrentPage(prev => prev + 1);
            fetchMovies(currentPage + 1);
        }
    };

    useEffect(() => {
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, [isLoading, hasMore]);

    // Search for a movie by title
    const handleSearch = (query) => {
        const filtered = movies.filter(movie =>
            movie.Title.toLowerCase().includes(query.toLowerCase())
        );
        setFilteredMovies(filtered);
    };

    // Sorting movies by year
    const handleSort = (order) => {
        const sorted = [...filteredMovies];
        if (order === 'newest') {
            sorted.sort((a, b) => b.Year - a.Year);
        } else if (order === 'oldest') {
            sorted.sort((a, b) => a.Year - b.Year);
        }
        setFilteredMovies(sorted);
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
                        />
                    </div>
                    <div className={`${style.listOfMovies} thq-grid-4`}>
                        {filteredMovies.map(movie => {
                            console.log('Rendering movie:', movie);
                            return (
                                <div key={movie.Movie_ID} className={style.movie}
                                     onClick={() => handleMovieClick(movie.Movie_ID)}>
                                    <img
                                        alt={movie.Title}
                                        src={movie.Poster}
                                        className={`${style.image} thq-img-ratio-16-9`}
                                    />
                                    <div className={style.movieTitle}>{movie.Title}</div>
                                </div>
                            );
                        })}
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