import React, { useState } from 'react'
import PropTypes from 'prop-types'

import style from './Filter.module.css'

const Filter = ({ onSearch, onSort, currentSort, sortOrder, sortOptions }) => {
    const [searchQuery, setSearchQuery] = useState('');

    const handleSearchChange = (e) => {
        const value = e.target.value;
        setSearchQuery(value);
        onSearch(value);
    };

    const handleSortChange = (field) => {
        onSort(field);
    };

    return (
        <div className={style.filterContainer}>
            <div className={style.searchContainer}>
                <input
                    type="text"
                    placeholder="Поиск фильмов..."
                    value={searchQuery}
                    onChange={handleSearchChange}
                    className={style.searchInput}
                />
            </div>
            <div className={style.sortContainer}>
                {sortOptions.map(option => (
                    <button
                        key={option.value}
                        className={`${style.sortButton} ${currentSort === option.value ? style.active : ''}`}
                        onClick={() => handleSortChange(option.value)}
                    >
                        {option.label}
                        {currentSort === option.value && (
                            <span className={style.sortOrder}>
                                {sortOrder === 'asc' ? ' ↑' : ' ↓'}
                            </span>
                        )}
                    </button>
                ))}
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
