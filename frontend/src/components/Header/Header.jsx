import React, {useState} from 'react';
import style from './Header.module.css';
import '../../style.css'
import hamburger from '../../data/hamburger1.png'
import {Link} from "react-router-dom";
import Login from "../Pages/Login/Login";
import { useAuth } from '../../contexts/AuthContext';

const Header = () => {
    const [menuOpen, setMenuOpen] = useState(false);
    const [isLoginOpen, setIsLoginOpen] = useState(false);
    const { currentUser } = useAuth();
    const toggleMenu = () => { setMenuOpen(!menuOpen);}

    return (
        <header className={style.header}>
            <header data-thq="thq-navbar" className={style.navbarInteractive}>
                <div className={style.branding}>
                    <Link to="/">
                        <svg width="24" height="24" viewBox="0 0 24 24" className={style.logo}>
                            <g fill="none" color="currentColor" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                                <circle r="8.5" cx="10.5" cy="10.5"></circle>
                                <path d="M10.5 10.5h.008M14 7l-1 1m-5 5l-1 1m7 0l-1-1M8 8L7 7m6.5 11.5l5.823-.965C20.719 17.292 22 18.35 22 19.75c0 1.243-1.021 2.25-2.281 2.25H18.7"></path>
                            </g>
                        </svg>
                    </Link>
                </div>
                <div className={style.items}>
                    <div className={style.links}>
                        <Link to="/" className={`${style.text} nav-link`}>Home</Link>
                        <Link to="/movies" className={`${style.text} nav-link`}>Movies</Link>
                    </div>
                    <div className={style.buttons}>
                        <button 
                            className={`${style.button} start-button button`}
                            onClick={() => setIsLoginOpen(true)}
                        >
                            <span className={style.text17}>
                                {currentUser ? currentUser.name : 'Sign In'}
                            </span>
                        </button>
                    </div>
                </div>

                <div data-thq="thq-burger-menu" className={style.burgerMenu}>
                    <button className={`${style.button2} button`} onClick={toggleMenu}>
                        <img alt="logo" src={hamburger} className={style.image}/>
                        <span className={style.text18}>Menu</span>
                    </button>
                </div>

                <div className={`${style.mobileMenu} ${menuOpen ? style.open : ''}`}>
                    <div className={style.nav}>
                        <div className={style.top}>
                            <div className={style.branding2}>
                                <Link to="/" className={style.company1} onClick={toggleMenu}>
                                    Cinema
                                </Link>
                            </div>
                            <div className={style.menuClose} onClick={toggleMenu}>
                                <svg width="24" height="24" viewBox="0 0 24 24" className={style.icon18}>
                                    <path d="M18 6L6 18M6 6l12 12"></path>
                                </svg>
                            </div>
                        </div>
                        <div className={style.items2}>
                            <div className={style.links2}>
                                <div className={style.container2}>
                                    <Link to="/" className="nav-link" onClick={toggleMenu}>Home</Link>
                                </div>
                                <div className={style.container2}>
                                    <Link to="/movies" className="nav-link" onClick={toggleMenu}>Movies</Link>
                                </div>
                            </div>
                        </div>
                        <div className={style.buttons2}>
                            <button 
                                className={`${style.button3} start-button button`}
                                onClick={() => {
                                    setIsLoginOpen(true);
                                    toggleMenu();
                                }}
                            >
                                <span className={style.text22}>
                                    {currentUser ? currentUser.name : 'Sign In'}
                                </span>
                            </button>
                        </div>
                    </div>
                </div>
            </header>
            <Login isOpen={isLoginOpen} onClose={() => setIsLoginOpen(false)} />
        </header>
    );
};

export default Header;