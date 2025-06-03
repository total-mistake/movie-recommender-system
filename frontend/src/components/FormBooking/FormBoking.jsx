import React, {useEffect, useState} from 'react';
import style from './FormBooking.module.css';
import {Link} from "react-router-dom";
import { useAuth } from '../../contexts/AuthContext';
import Login from '../Pages/Login/Login';

const FormBoking = ({ movieId }) => {
    const { currentUser } = useAuth();
    const [movies, setMovies] = useState([]);
    const [sessions, setSessions] = useState([]);
    const [selectedMovie, setSelectedMovie] = useState(movieId);
    const [selectedSession, setSelectedSession] = useState('');
    const [isLoginOpen, setIsLoginOpen] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const [numTickets, setNumTickets] = useState(1);
    const [isSubmitted, setIsSubmitted] = useState(false);

    // Загрузка списка фильмов
    useEffect(() => {
        fetch('/api/movies')
            .then(response => response.json())
            .then(data => {
                setMovies(data);
                if (movieId) {
                    setSelectedMovie(movieId);
                    // Загружаем сеансы для выбранного фильма
                    fetch(`/api/sessions/movie/${movieId}`)
                        .then(response => response.json())
                        .then(sessionsData => {
                            console.log('Sessions data:', sessionsData);
                            setSessions(sessionsData);
                        })
                        .catch(error => {
                            console.error('Error fetching sessions:', error);
                            setError('Failed to load sessions. Please try again later.');
                        });
                }
            })
            .catch(error => {
                console.error('Error fetching movies:', error);
                setError('Failed to load movies. Please try again later.');
            });
    }, [movieId]);

    // Загрузка сеансов при изменении выбранного фильма
    useEffect(() => {
        if (selectedMovie) {
            console.log('Fetching sessions for movie:', selectedMovie);
            fetch(`/api/sessions/movie/${selectedMovie}`)
                .then(response => response.json())
                .then(data => {
                    console.log('Sessions data:', data);
                    setSessions(data);
                    setSelectedSession(''); // Сбрасываем выбранный сеанс
                })
                .catch(error => {
                    console.error('Error fetching sessions:', error);
                    setError('Failed to load sessions. Please try again later.');
                });
        }
    }, [selectedMovie]);

    const resetForm = () => {
        setSelectedMovie('');
        setSelectedSession('');
        setNumTickets(1);
        setError('');
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        setError('');

        if (!selectedSession) {
            setError('Please select a session');
            setIsLoading(false);
            return;
        }

        console.log('Selected session ID:', selectedSession);
        console.log('All sessions:', sessions);
        const selectedSessionData = sessions.find(s => s.id === selectedSession);
        console.log('Found session data:', selectedSessionData);

        if (!selectedSessionData) {
            setError('Selected session not found');
            setIsLoading(false);
            return;
        }

        const ticket = {
            session: { id: selectedSession },
            numTickets: parseInt(numTickets),
            totalPrice: selectedSessionData.price * parseInt(numTickets)
        };

        console.log('Sending ticket data:', ticket);

        try {
            const response = await fetch('http://localhost:8080/api/tickets', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'User-Id': currentUser.id
                },
                credentials: 'include',
                body: JSON.stringify(ticket)
            });

            if (response.ok) {
                setIsSubmitted(true);
                resetForm();
            } else {
                const errorData = await response.json().catch(() => ({}));
                if (response.status === 401) {
                    setError('Your session has expired. Please sign in again.');
                } else if (response.status === 403) {
                    setError('You do not have permission to book tickets. Please sign in.');
                } else {
                    setError(errorData.message || 'Failed to book ticket. Please try again later.');
                }
            }
        } catch (error) {
            console.error('Error booking ticket:', error);
            setError('Failed to book ticket. Please try again later.');
        } finally {
            setIsLoading(false);
        }
    };

    if (!currentUser) {
        return (
            <div className={style.container1}>
                <div className={style.purchaseTicket}>
                    <div className={style.authMessage}>
                        <p>Please sign in to purchase a ticket</p>
                        <button 
                            className={`${style.button} thq-button-filled`}
                            onClick={() => setIsLoginOpen(true)}
                        >
                            <span className={style.text19}>Sign In</span>
                        </button>
                    </div>
                    <Login isOpen={isLoginOpen} onClose={() => setIsLoginOpen(false)} />
                </div>
            </div>
        );
    }

    return (
        <div className={style.container1}>
            <div className={style.purchaseTicket}>
                <h1 className={`${style.text10} ${style.text18}`}>
                    Ticket purchase
                </h1>
                <form className={style.container2} onSubmit={handleSubmit}>
                    <div className={style.movie1}>
                        <label htmlFor="movie-select" className={style.text11}>
                            Select Movie:
                        </label>
                        <select
                            id="movie-select"
                            className={`${style.select1} thq-input`}
                            value={selectedMovie}
                            onChange={(e) => setSelectedMovie(e.target.value)}
                            required
                        >
                            <option value="">Select a movie</option>
                            {movies.map(movie => (
                                <option key={movie.id} value={movie.id}>
                                    {movie.title}
                                </option>
                            ))}
                        </select>
                    </div>

                    {selectedMovie && (
                        <div className={style.movie1}>
                            <label htmlFor="session-select" className={style.text11}>
                                Select Session:
                            </label>
                            <select
                                id="session-select"
                                className={`${style.select1} thq-input`}
                                value={selectedSession}
                                onChange={(e) => setSelectedSession(e.target.value)}
                                required
                            >
                                <option value="">Select a session</option>
                                {sessions.map(session => (
                                    <option key={session.id} value={session.id}>
                                        {new Date(session.date).toLocaleDateString()} at {session.time}
                                    </option>
                                ))}
                            </select>
                        </div>
                    )}

                    {selectedSession && (
                        <div className={style.movie2}>
                            <div className={style.container4}>
                                <label htmlFor="num-tickets" className={style.text16}>
                                    Number of tickets
                                </label>
                                <select
                                    id="num-tickets"
                                    className={`${style.select2} thq-input`}
                                    value={numTickets}
                                    onChange={(e) => setNumTickets(e.target.value)}
                                    required
                                >
                                    <option value="1">1</option>
                                    <option value="2">2</option>
                                    <option value="3">3</option>
                                </select>
                            </div>
                            <h1 className={style.text17}>
                                <span className={style.text20}>
                                    Total: <span className={style.text22}>
                                        {(() => {
                                            const session = sessions.find(s => s.id === selectedSession);
                                            console.log('Session for price calculation:', session);
                                            return `$${session?.price * numTickets || 0}`;
                                        })()}
                                    </span>
                                </span>
                            </h1>
                        </div>
                    )}

                    {error && <div className={style.error}>{error}</div>}
                    
                    {isSubmitted ? (
                        <>
                            <div className={style.sendtext}>
                                Your ticket has been successfully booked. We look forward to seeing you at the movie!
                            </div>
                            <Link to={"/movies"} className={style.link}>
                                <button className={`${style.button2} start-button button`}>
                                    <span className={style.text21}>back to movies</span>
                                </button>
                            </Link>
                        </>
                    ) : (
                        <button 
                            type="submit" 
                            className={`${style.button} thq-button-filled`}
                            disabled={!selectedSession || isLoading}
                        >
                            <span className={style.buttontext}>
                                {isLoading ? 'Booking...' : 'Book Tickets'}
                            </span>
                        </button>
                    )}
                </form>
            </div>
        </div>
    );
};

export default FormBoking;