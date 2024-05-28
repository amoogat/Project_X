import axios from 'axios';

const API = axios.create({
    baseURL: 'http://localhost:8000/api/',  // Django backend URL
});

export const getCsrfToken = () => {
    const csrfToken = document.cookie.split(';')
        .find(cookie => cookie.trim().startsWith('csrftoken='))
        ?.split('=')[1];
    return csrfToken;
};

API.interceptors.request.use(config => {
    const csrfToken = getCsrfToken();
    if (csrfToken) {
        config.headers['X-CSRFToken'] = csrfToken;
    }
    return config;
});

export default API;
