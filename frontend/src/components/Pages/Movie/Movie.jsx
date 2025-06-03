import React, {useEffect, useState} from 'react';
import Header from "../../Header/Header";
import Footer from "../../Footer/Footer";
import AboutMovie from "../../AboutMovie/AboutMovie";
import ContactForm from "../../ContactForm/ContactForm";
import './Movie.css'
import Reviews from "../../Reviews/Reviews";
import {useParams} from "react-router-dom";
import { movieService } from '../../../../services/movieService';

const Movie = () => {
    const { id } = useParams();
    const [movie, setMovie] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchMovie = async () => {
          try {
            setIsLoading(true);
            setError(null);
            const data = await movieService.getMovie(id);
            setMovie(data);
          } catch (err) {
            setError(err.message);
            console.error('Ошибка при загрузке фильма:', err);
          } finally {
            setIsLoading(false);
          }
        };

        fetchMovie();
    }, [id]);

    if (isLoading) {
        return <div>Загрузка данных о фильме...</div>;
    }

    if (error) {
        return <div>Ошибка: {error}</div>;
    }

    if (!movie) {
        return <div>Фильм не найден</div>;
    }

    return (
        <body>
            <Header/>
            <div className="movies">
                {movie && <AboutMovie movie={movie} />}
                <div className="aboutmovie">
                    <ContactForm
                        rootClassName="rootClassName"
                        id={movie.Movie_ID}
                    />
                    <div className="aboutmoviepicture"></div>
                </div>
                {movie && <Reviews id={movie.Movie_ID} />}
            </div>
            <Footer/>
        </body>
    );
};

export default Movie;

