import { createApp } from 'vue';
import App from './App.vue';
import { createRouter, createWebHistory } from 'vue-router';
import TrainingPage from './components/TrainingPage.vue';
import PredictPage from './components/PredictPage.vue';

const routes = [
  { path: '/', redirect: '/train' },
  { path: '/train', component: TrainingPage },
  { path: '/predict', component: PredictPage },
];

const router = createRouter({ history: createWebHistory(), routes });
createApp(App).use(router).mount('#app');
