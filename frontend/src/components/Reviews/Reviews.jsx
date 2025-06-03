import React, { useState, useEffect } from 'react';
import style from './Reviews.module.css';
import Review from "../Review/Review";

// Компонент для отображения списка отзывов к фильму
const Reviews = ({ id }) => {
    // Состояние для хранения списка отзывов
    const [reviews, setReviews] = useState([]);
    
    // Эффект для загрузки отзывов при монтировании компонента или изменении id фильма
    useEffect(() => {
        // Запрос к API для получения отзывов конкретного фильма
        fetch(`/api/reviews/by-film/${id}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                setReviews(data);
            })
            .catch(error => {
                console.error('Error fetching reviews:', error);
            });
    }, [id]);

    return (
        <div className="thq-section-padding">
            <div className={style.maxWidth}>
                <div className={`${style.container2} thq-grid-2`}>
                    {/* Отображение каждого отзыва в виде карточки */}
                    {reviews.map(review => (
                        <Review
                            key={review.id}
                            name={review.user?.name || 'Unknown User'}
                            rating={Number(review.rating)}
                            text={review.reviewText}
                            date={review.reviewDate}
                        />
                    ))}
                </div>
            </div>
        </div>
    );
};

export default Reviews;
