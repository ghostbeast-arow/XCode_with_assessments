import { createApp } from 'vue';
import { createPinia } from 'pinia';
import App from './App.vue';
import 'maplibre-gl/dist/maplibre-gl.css';
import './styles/main.scss';

const app = createApp(App);
app.use(createPinia());
app.mount('#app');
