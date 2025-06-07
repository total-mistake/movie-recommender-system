import React, {useEffect, useState} from 'react';
import Header from "../../Header/Header";
import Footer from "../../Footer/Footer";
import {useParams} from "react-router-dom";
import { fetchMovieById } from '../../../services/movieService';
import style from './Movie.css';

const Movie = () => {
    const { id } = useParams();
    const [movie, setMovie] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const loadMovie = async () => {
            try {
                setIsLoading(true);
                setError(null);
                const data = await fetchMovieById(id);
                setMovie(data);
            } catch (err) {
                setError(err.message);
                console.error('Ошибка при загрузке фильма:', err);
            } finally {
                setIsLoading(false);
            }
        };

        loadMovie();
    }, [id]);

    if (isLoading) {
        return (
            <div className={style.loadingContainer}>
                <div className={style.loading}>Загрузка данных о фильме...</div>
            </div>
        );
    }

    if (error) {
        return (
            <div className={style.errorContainer}>
                <div className={style.error}>Ошибка: {error}</div>
            </div>
        );
    }

    if (!movie) {
        return (
            <div className={style.errorContainer}>
                <div className={style.error}>Фильм не найден</div>
            </div>
        );
    }

    return (
        <body>
            <Header/>
            <div className={style.movieContainer}>
                <div className={style.movieContent}>
                    <div className={style.movieHeader}>
                        <div className={style.posterContainer}>
                            <img 
                                src={movie.Poster} 
                                alt={movie.Title} 
                                className={style.poster}
                            />
                        </div>
                        <div className={style.movieInfo}>
                            <h1 className={style.title}>{movie.Title}</h1>
                            <div className={style.year}>{movie.Year}</div>
                            
                            <div className={style.ratingContainer}>
                                <div className={style.rating}>
                                    <span className={style.ratingValue}>{movie.Rating.toFixed(1)}</span>
                                    <span className={style.ratingCount}>({movie.Rating_Count} оценок)</span>
                                </div>
                            </div>

                            <div className={style.details}>
                                <div className={style.detailItem}>
                                    <span className={style.detailLabel}>Жанры:</span>
                                    <span className={style.detailValue}>{movie.Genres}</span>
                                </div>
                                <div className={style.detailItem}>
                                    <span className={style.detailLabel}>Страна:</span>
                                    <span className={style.detailValue}>{movie.Countries}</span>
                                </div>
                                <div className={style.detailItem}>
                                    <span className={style.detailLabel}>Режиссер:</span>
                                    <span className={style.detailValue}>{movie.Directors}</span>
                                </div>
                                <div className={style.detailItem}>
                                    <span className={style.detailLabel}>Сценаристы:</span>
                                    <span className={style.detailValue}>{movie.Writers}</span>
                                </div>
                                <div className={style.detailItem}>
                                    <span className={style.detailLabel}>Актеры:</span>
                                    <span className={style.detailValue}>{movie.Actors}</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className={style.plotSection}>
                        <h2 className={style.plotTitle}>Описание</h2>
                        <p className={style.plot}>{movie.Plot}</p>
                    </div>
                </div>
            </div>
            <Footer/>
        </body>
    );
};

export default Movie;

