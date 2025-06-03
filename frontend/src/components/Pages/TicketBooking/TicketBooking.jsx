import React from 'react';
import Header from "../../Header/Header";
import Footer from "../../Footer/Footer";
import FormBoking from "../../FormBooking/FormBoking";
import {useLocation} from "react-router-dom";

const TicketBooking = () => {
    const location = useLocation();
    const params = new URLSearchParams(location.search);
    const movieId = params.get('id');
    console.log('TicketBooking - movieId:', movieId);

    return (
        <body>
            <Header/>
            <FormBoking movieId={movieId}/>
            <Footer/>
        </body>
    );
};

export default TicketBooking;