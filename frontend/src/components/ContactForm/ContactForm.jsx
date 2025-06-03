import React, { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import style from './ContactForm.module.css';
import Login from '../Pages/Login/Login';

// Компонент формы для добавления нового отзыва
const ContactForm = ({ rootClassName, id, onReviewSubmitted }) => {
    const { currentUser } = useAuth();
    const [rating, setRating] = useState('');
    const [reviewText, setReviewText] = useState('');
    const [error, setError] = useState('');
    const [isSubmitted, setIsSubmitted] = useState(false);
    const [isLoginOpen, setIsLoginOpen] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!currentUser) {
            setError('Please sign in to leave a review');
            return;
        }

        console.log('Submitting review for movie:', id);
        console.log('Current user:', currentUser);
        console.log('User ID:', currentUser.id);

        try {
            const reviewData = {
                movie: { id: id.toString() },
                rating: parseInt(rating),
                reviewText,
                reviewDate: new Date().toISOString().split('T')[0]
            };
            console.log('Review data:', reviewData);

            const response = await fetch('http://localhost:8080/api/reviews', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'User-Id': currentUser.id.$oid || currentUser.id
                },
                credentials: 'include',
                body: JSON.stringify(reviewData)
            });

            console.log('Response status:', response.status);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                console.error('Error response:', errorData);
                if (response.status === 401) {
                    setError('Your session has expired. Please sign in again.');
                    return;
                }
                setError('Failed to submit review. Please try again later.');
                return;
            }

            const responseData = await response.json();
            console.log('Success response:', responseData);

            setIsSubmitted(true);
            setRating('');
            setReviewText('');
            if (onReviewSubmitted) {
                onReviewSubmitted();
            }
        } catch (error) {
            console.error('Error submitting review:', error);
            setError('Failed to submit review. Please try again later.');
        }
    };

    if (!currentUser) {
        return (
            <div className={style.contact5}>
                <div className={style.authMessage}>
                    <p>Please sign in to leave a review</p>
                    <button 
                        className={`${style.button} thq-button-filled`}
                        onClick={() => setIsLoginOpen(true)}
                    >
                        <span className={style.buttontext}>Sign In</span>
                    </button>
                </div>
                <Login isOpen={isLoginOpen} onClose={() => setIsLoginOpen(false)} />
            </div>
        );
    }

    return (
        <div className={`${style.contact5} ${rootClassName}`}>
            <span className={`${style.text1} thq-body-small`}>
                Share your thoughts about the movie with us
            </span>
            <form className={style.form} onSubmit={handleSubmit}>
                <div className={style.container}>
                    <div className={style.input2}>
                        <label htmlFor="contact-form-2-rating"
                               className={`${style.text3} thq-body-small`}>
                            Rate
                        </label>
                        <select
                            id="contact-form-2-rating"
                            className={`${style.select} thq-input`}
                            value={rating}
                            onChange={(e) => setRating(e.target.value)}
                            required
                        >
                            <option value="" disabled>Select Rating</option>
                            <option value="1">1</option>
                            <option value="2">2</option>
                            <option value="3">3</option>
                            <option value="4">4</option>
                            <option value="5">5</option>
                        </select>
                    </div>
                </div>

                <div className={style.input3}>
                    <textarea
                        id="contact-form-2-message"
                        rows="3"
                        placeholder="Share your opinion about the movie"
                        className={`${style.textarea} thq-input`}
                        value={reviewText}
                        onChange={(e) => setReviewText(e.target.value)}
                        required
                    ></textarea>
                </div>

                {error && <div className={style.error}>{error}</div>}

                {isSubmitted ? (
                    <div className={style.sendtext}>
                        Your review has been successfully submitted!
                    </div>
                ) : (
                    <button type="submit" className={`${style.button} thq-button-filled`}>
                        <span className={style.buttontext}>Send</span>
                    </button>
                )}
            </form>
        </div>
    );
};

export default ContactForm;