<template>
  <div class="border p-4 rounded bg-white shadow">
    <input v-model="className" placeholder="ชื่อคลาส" class="mb-2 p-1 border rounded w-full" />
    <div class="flex gap-2 mb-2">
      <button @click="openWebcam" class="px-2 py-1 bg-blue-100 rounded">📷 เว็บแคม</button>
      <input type="file" @change="handleUpload" />
    </div>
    <div v-if="images.length" class="mt-2 flex flex-wrap gap-1">
      <img v-for="(img, i) in images" :key="i" :src="img" class="w-16 h-16 object-cover rounded" />
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';

const className = ref('Class');
const images = ref([]);

function handleUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  images.value.push(URL.createObjectURL(file));
}

function openWebcam() {
  alert('(ฟีเจอร์เปิดกล้องจะเพิ่มภายหลัง)');
}
</script>
