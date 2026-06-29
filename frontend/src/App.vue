// Vue 3 + Vite + Webcam + Upload Skeleton for Sign Language Recognition

<script setup>
import { RouterLink, RouterView } from 'vue-router'

</script>

<template>
  <nav class="p-4 bg-gray-100">
    <router-link to="/train" class="mr-4">Train</router-link>
    <router-link to="/predict">Predict</router-link>
  </nav>
  <router-view />
</template>

<style>
</style>
