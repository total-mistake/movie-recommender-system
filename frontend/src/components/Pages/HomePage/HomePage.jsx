import React from 'react';
import style from './HomePage.module.css';
import Header from "../../Header/Header";
import Footer from "../../Footer/Footer";
import '../../../style.css'
import Gallery from "../../Gallery/Gallery";

const HomePage = () => {
    return (
        <body>
            <title>Films - main</title>
            <meta property="og:title" content="Films - main"/>
            <Header/>

            <header className={style.hero}>
                <div className={style.header2}>
                    <h1 className={style.title}>Discover Your Next Favorite Movie</h1>
                    <p className={style.description1}>
                        Get personalized movie recommendations based on your preferences
                        and explore a vast collection of films across all genres
                    </p>
                </div>
            </header>

            <div className={style.about}>
                <div className={style.header3}>
                    <div className={style.container5}>
                        <div className={style.container6}>
                            <h2 className={style.company2}>Smart Movie Discovery</h2>
                            <h1 className={style.text23}>
                                Your Personal Movie Guide
                            </h1>
                        </div>
                        <span className={style.description2}>
                            Welcome to Movie Recommender, your intelligent companion for discovering films. 
                            Our advanced recommendation system analyzes your preferences and viewing history 
                            to suggest movies you'll love. Browse through our extensive collection, read 
                            reviews from fellow movie enthusiasts, and find your next cinematic adventure. 
                            Whether you're into classic films, indie gems, or the latest blockbusters, 
                            we'll help you find the perfect movie for any mood or occasion.
                        </span>
                    </div>
                </div>
            </div>

            <Gallery
                text={"RECOMMENDED MOVIES"}
                content1={"Explore our curated selection of recommended movies, carefully chosen based on your preferences and viewing history."}
                heading1={"Personalized Movie Recommendations"}
            ></Gallery>

            <Footer/>
        </body>
    )
};

export default HomePage;