import React from 'react';
import style from './Gallery.module.css'
import {Link} from "react-router-dom";


const Gallery = (props) => {
    return (
        <div className={`${style.gallery3} thq-section-padding`}>
            <div className={style.maxWidth}>
                <div className={style.sectionTitle}>
                    <h2 className={`${style.text1} ${style.text6} thq-heading-2`}>
                        {props.heading1 ? props.heading1 : (
                            "Explore Our movie Gallery"
                        )}
                    </h2>
                    <span className={`${style.text2} ${style.text5} thq-body-large`}>
                        {props.content1 ? props.content1 : (
                              "Browse through a stunning collection of movie posters from various genres and eras. Find your favorites and get ready for an immersive cinema experience."
                        )}
                    </span>
                </div>
                <div className={style.container1}>
                    <div className={style.content}>
                        <div className={style.container2}>
                            <img
                                alt={props.image1Alt}
                                src="https://i.pinimg.com/736x/6a/f5/a1/6af5a1c2a2c00f55a216dd9a74e1e1cc.jpg"
                                className={`${style.image1} thq-img-ratio-1-1`}
                            />
                            <img
                                alt={props.image2Alt}
                                src="https://avatars.mds.yandex.net/get-kinopoisk-image/1704946/57d9492f-84a9-4749-aeb1-d172d60c7793/1920x"
                                className={`${style.image2} thq-img-ratio-1-1`}
                            />
                        </div>
                        <div className={style.container3}>
                            <img
                                alt={props.image3Alt}
                                src="https://images.kinorium.com/movie/poster/472809/w1500_44943050.jpg"
                                className={`${style.image3} thq-img-ratio-4-3`}
                            />
                            <Link to={"/movies"} className={style.linkB}>
                            <button className={`${style.button} start-button button`}>
                                <span className={style.text3}>
                                  {props.text ? props.text : ("Buy a ticket")}
                                </span>
                            </button>
                            </Link>
                            <img
                                alt={props.image4Alt}
                                src="https://avatars.dzeninfra.ru/get-zen_doc/751940/pub_5e7d004d6c402b45fcb1649d_5e7d01918b63b9743c9ae4df/scale_1200"
                                className={`${style.image4} thq-img-ratio-1-1`}
                            />
                            <img
                                alt={props.image5Alt}
                                src="https://cdn1.ozone.ru/s3/multimedia-t/6292258685.jpg"
                                className={`${style.image5} thq-img-ratio-4-3`}
                            />
                        </div>
                        <div className={style.container4}>
                            <img
                                alt={props.image6Alt}
                                src="https://cdn.ananasposter.ru/image/cache/catalog/poster/pos23/29/70462-1000x830.jpg"
                                className={`${style.image6} thq-img-ratio-1-1`}
                            />
                            <img
                                alt={props.image7Alt}
                                src="https://cdn1.ozone.ru/s3/multimedia-f/6667782603.jpg"
                                className={`${style.image7} thq-img-ratio-1-1`}
                            />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

Gallery.defaultProps = {
    image3Alt: 'movie.js Poster 3',
    image7Alt: 'movie.js Poster 7',
    image1Alt: 'movie.js Poster 1',
    text: undefined,
    image6Alt: 'movie.js Poster 6',
    content1: undefined,
    image4Alt: 'movie.js Poster 4',
    image5Alt: 'movie.js Poster 5',
    heading1: undefined,
    image2Alt: 'movie.js Poster 2',
}

export default Gallery;