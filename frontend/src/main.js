import { createApp } from 'vue';
import App from './App.vue';
import { createRouter, createWebHistory } from 'vue-router';
import TrainingPage from './components/TrainingPage.vue';
import PredictPage from './components/PredictPage.vue';
import ExpTrainingPage from './components/exp/ExpTrainingPage.vue';
import ExpPredictPage from './components/exp/ExpPredictPage.vue';

const routes = [
  { path: '/', redirect: '/train' },
  { path: '/train', component: TrainingPage },
  { path: '/predict', component: PredictPage },
  { path: '/exp', redirect: '/exp/predict' },
  { path: '/exp/train', component: ExpTrainingPage },
  { path: '/exp/predict', component: ExpPredictPage },
];

const router = createRouter({ history: createWebHistory(), routes });
createApp(App).use(router).mount('#app');
