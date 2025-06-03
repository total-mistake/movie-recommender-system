import React from 'react';
import style from './AboutMovie.module.css'
import PropTypes from "prop-types";
import {Link} from "react-router-dom";

const AboutMovie = ({movie}) => {
    return (
        <div className={`${style.container10} thq-section-padding`}>
            <div className={`${style.maxWidth} thq-section-max-width`}>
                <div
                    className={style.poster}
                    style={{ backgroundImage: `url(${movie.poster})` }}
                />
                <div className={`${style.container12} thq-flex-column`}>
                    <h2 className={`${style.text10} thq-heading-2`}>
                        {movie.title ?? ("Film Name")}
                    </h2>
                    <p className={`${style.text11} thq-body-large`}>
                        Rating: {movie.rating ?? ("4.5")}
                    </p>
                    <div className={`${style.container13} thq-grid-2`}>
                        <div className={style.containerField}>
                            <h2 className={style.title}>Directed by</h2>
                            <span className={style.underTitle}>
                                {movie.director ?? ("Total films in our database")}
                              </span>
                        </div>
                        <div className={style.containerField}>
                            <h2 className={style.title}>Genres</h2>
                            <span className={style.underTitle}>
                                {movie.genre ?? ("Based on user reviews")}
                              </span>
                        </div>
                    </div>
                    <div className={`${style.container13} thq-grid-2`}>
                        <div className={style.containerField}>
                            <h2 className={style.title}>Duration</h2>
                            <span className={style.underTitle}>
                                {movie.duration ?? ("100")}
                              </span>
                        </div>
                        <div className={style.containerField}>
                            <h2 className={style.title}>Year of creation</h2>
                            <span className={style.underTitle}>
                                {movie.year ?? ("2000")}
                              </span>
                        </div>
                    </div>

                    <div className={style.container17}>
                        <h2 className={`${style.text16} thq-heading-2`}>Description</h2>
                        <span className={style.description}>
                            {movie.description ?? ("Some description of the movie")}
                          </span>
                    </div>

                    <div className={style.container20}>
                        <Link to={`/ticket?id=${movie.id}`} className={style.link}>
                            <button className={`${style.button1} start-button button`}>
                              <span className={style.text20}>Buy a ticket </span>
                            </button>
                        </Link>
                        <Link to={"/movies"} className={style.link}>
                            <button className={`${style.button2} start-button button`}>
                              <span className={style.text21}>back to movies</span>
                            </button>
                        </Link>
                    </div>
                </div>
            </div>
        </div>
    );
};

AboutMovie.propTypes = {
    movie: PropTypes.shape({
        id: PropTypes.string.isRequired,
        title: PropTypes.string.isRequired,
        rating: PropTypes.number.isRequired,
        director: PropTypes.string.isRequired,
        genre: PropTypes.string.isRequired,
        description: PropTypes.string.isRequired
    }).isRequired
};

export default AboutMovie;