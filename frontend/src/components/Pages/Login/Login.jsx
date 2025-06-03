import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Formik, Form, Field, ErrorMessage } from 'formik';
import * as Yup from 'yup';
import style from './Login.module.css';
import { useAuth } from '../../../contexts/AuthContext';

const Login = ({ isOpen, onClose }) => {
    const navigate = useNavigate();
    const { currentUser, login, register, logout } = useAuth();
    const [isLogin, setIsLogin] = useState(true);
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    // Схема валидации для входа
    const loginSchema = Yup.object().shape({
        username: Yup.string()
            .min(3, 'Username must be at least 3 characters')
            .required('Username is required'),
        password: Yup.string()
            .required('Password is required')
    });

    // Схема валидации для регистрации
    const registerSchema = Yup.object().shape({
        name: Yup.string()
            .min(2, 'Name must be at least 2 characters')
            .max(50, 'Name must be less than 50 characters')
            .required('Name is required'),
        username: Yup.string()
            .min(3, 'Username must be at least 3 characters')
            .required('Username is required'),
        password: Yup.string()
            .min(6, 'Password must be at least 6 characters')
            .matches(/[A-Z]/, 'Password must contain at least one uppercase letter')
            .matches(/[a-z]/, 'Password must contain at least one lowercase letter')
            .matches(/[0-9]/, 'Password must contain at least one number')
            .required('Password is required')
    });

    const handleSubmit = async (values, { setSubmitting }) => {
        setError('');
        setIsLoading(true);

        try {
            if (isLogin) {
                await login(values);
                onClose();
                navigate('/');
            } else {
                await register(values);
                onClose();
                navigate('/');
            }
        } catch (error) {
            setError(error.message);
        } finally {
            setIsLoading(false);
            setSubmitting(false);
        }
    };

    const handleLogout = () => {
        logout();
        onClose();
        navigate('/');
    };

    if (!isOpen) return null;

    if (currentUser) {
        return (
            <div className={style.modalOverlay} onClick={onClose}>
                <div className={style.modalContent} onClick={e => e.stopPropagation()}>
                    <button className={style.closeButton} onClick={onClose}>×</button>
                    <div className={style.formContainer}>
                        <h1 className={style.title}>Profile</h1>
                        <div className={style.profileInfo}>
                            <div className={style.profileField}>
                                <label className={style.label}>Name:</label>
                                <span className={style.profileValue}>{currentUser.name}</span>
                            </div>
                            <div className={style.profileField}>
                                <label className={style.label}>Username:</label>
                                <span className={style.profileValue}>{currentUser.username}</span>
                            </div>
                        </div>
                        <button 
                            onClick={handleLogout}
                            className={`${style.button} ${style.logoutButton}`}
                        >
                            <span className={style.buttonText}>Logout</span>
                        </button>
                    </div>
                </div>
            </div>
        );
    }

    const initialValues = isLogin 
        ? { username: '', password: '' }
        : { name: '', username: '', password: '' };

    return (
        <div className={style.modalOverlay} onClick={onClose}>
            <div className={style.modalContent} onClick={e => e.stopPropagation()}>
                <button className={style.closeButton} onClick={onClose}>×</button>
                <div className={style.formContainer}>
                    <div className={style.tabs}>
                        <button 
                            className={`${style.tab} ${isLogin ? style.activeTab : ''}`}
                            onClick={() => setIsLogin(true)}
                        >
                            Sign In
                        </button>
                        <button 
                            className={`${style.tab} ${!isLogin ? style.activeTab : ''}`}
                            onClick={() => setIsLogin(false)}
                        >
                            Register
                        </button>
                    </div>
                    <h1 className={style.title}>{isLogin ? 'Sign In' : 'Register'}</h1>
                    
                    <Formik
                        initialValues={initialValues}
                        validationSchema={isLogin ? loginSchema : registerSchema}
                        onSubmit={handleSubmit}
                    >
                        {({ isSubmitting, touched, errors }) => (
                            <Form className={style.form}>
                                {!isLogin && (
                                    <div className={style.inputGroup}>
                                        <label htmlFor="name" className={style.label}>
                                            Name
                                        </label>
                                        <Field
                                            type="text"
                                            id="name"
                                            name="name"
                                            className={`${style.input} ${touched.name && errors.name ? style.inputError : ''}`}
                                        />
                                        <ErrorMessage name="name" component="span" className={style.errorText} />
                                    </div>
                                )}
                                <div className={style.inputGroup}>
                                    <label htmlFor="username" className={style.label}>
                                        Username
                                    </label>
                                    <Field
                                        type="text"
                                        id="username"
                                        name="username"
                                        className={`${style.input} ${touched.username && errors.username ? style.inputError : ''}`}
                                    />
                                    <ErrorMessage name="username" component="span" className={style.errorText} />
                                </div>
                                <div className={style.inputGroup}>
                                    <label htmlFor="password" className={style.label}>
                                        Password
                                    </label>
                                    <Field
                                        type="password"
                                        id="password"
                                        name="password"
                                        className={`${style.input} ${touched.password && errors.password ? style.inputError : ''}`}
                                    />
                                    <ErrorMessage name="password" component="span" className={style.errorText} />
                                </div>
                                {error && <div className={style.error}>{error}</div>}
                                <button 
                                    type="submit" 
                                    className={`${style.button} start-button button`}
                                    disabled={isSubmitting || isLoading}
                                >
                                    <span className={style.buttonText}>
                                        {isLoading ? 'Loading...' : (isLogin ? 'Sign In' : 'Register')}
                                    </span>
                                </button>
                            </Form>
                        )}
                    </Formik>
                </div>
            </div>
        </div>
    );
};

export default Login; 