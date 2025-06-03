import React from 'react'
import PropTypes from 'prop-types'

import style from './Filter.module.css'

const Filter = ({ onSearch, onSort }) => {
    return (
        <div className={`item ${style.sectionTitle}`}>
            <div className={`${style.cont1} service`}>
                <div className={style.details}>
                    <span className={style.text1}>Search BY</span>
                    <input
                        type="text"
                        placeholder="Enter a movie title..."
                        onChange={(e) => onSearch(e.target.value)}
                        className={style.searchInput}
                    />
                </div>
                <svg width="32" height="32" viewBox="0 0 32 32" className={style.icon1}>
                    <path
                        d="m29 27.586l-7.552-7.552a11.018 11.018 0 1 0-1.414 1.414L27.586 29ZM4 13a9 9 0 1 1 9 9a9.01 9.01 0 0 1-9-9"
                        fill="currentColor"
                    ></path>
                </svg>
            </div>

            <div className={`${style.cont1} service`}>
                <div className={style.details}>
                    <span className={style.text1}>Sort by year</span>
                    {/* Сортировка по году */}
                    <select
                        onChange={(e) => onSort(e.target.value)}
                        className={style.sortSelect}
                    >
                        <option value="">Select</option>
                        <option value="newest">Newest first</option>
                        <option value="oldest">Oldest first</option>
                    </select>
                </div>
                <svg viewBox="0 0 1024 1024" className={style.icon1}>
                    <path
                        d="M298.667 341.333h323.669l-353.835 353.835c-16.683 16.683-16.683 43.691 0 60.331s43.691 16.683 60.331 0l353.835-353.835v323.669c0 23.552 19.115 42.667 42.667 42.667s42.667-19.115 42.667-42.667v-426.667c0-5.803-1.152-11.307-3.243-16.341s-5.163-9.728-9.216-13.781c-0.043-0.043-0.043-0.043-0.085-0.085-3.925-3.925-8.619-7.083-13.781-9.216-5.035-2.091-10.539-3.243-16.341-3.243h-426.667c-23.552 0-42.667 19.115-42.667 42.667s19.115 42.667 42.667 42.667z"></path>
                </svg>
            </div>
        </div>
    )
}

Filter.defaultProps = {
    rootClassName: '',
}

Filter.propTypes = {
    rootClassName: PropTypes.string,
}

export default Filter
