import React from 'react';
import PropTypes from 'prop-types';
import style from './Review.module.css';

// Компонент для отображения отдельного отзыва
const Review = ({ name, rating, text, date }) => {
    return (
        <div className={`${style.card} thq-card`}>
            {/* Верхняя часть карточки с аватаром и информацией */}
            <div className={style.top}>
                <div className={style.container2}>
                    {/* Имя пользователя и дата отзыва */}
                    <div className={style.container3}>
                        <strong className={style.name}>{name}</strong>
                        <span className={style.date}>{date}</span>
                    </div>

                    {/* Отображение рейтинга в виде звезд */}
                    <div className={style.rating}>
                        {[...Array(5)].map((_, index) => (
                            <span 
                                key={index} 
                                className={`${style.star} ${index < rating ? style.filled : ''}`}
                            >
                                ★
                            </span>
                        ))}
                    </div>
                </div>
            </div>
            {/* Текст отзыва */}
            <div className={style.details}>
                <p className={style.text}>{text}</p>
            </div>
        </div>
    );
};

// Определение типов пропсов для компонента
Review.propTypes = {
    name: PropTypes.string.isRequired,
    rating: PropTypes.number.isRequired,
    text: PropTypes.string.isRequired,
    date: PropTypes.string.isRequired
};

export default Review;
