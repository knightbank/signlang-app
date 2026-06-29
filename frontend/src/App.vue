<script setup>
import { RouterLink, RouterView, useRoute } from 'vue-router'
import { computed } from 'vue'
const route = useRoute()
const isExp = computed(() => route.path.startsWith('/exp'))
</script>

<template>
  <nav :class="['app-nav', { exp: isExp }]">
    <span v-if="isExp" class="exp-tag">🧪 exp</span>
    <router-link :to="isExp ? '/exp/train' : '/train'" class="nav-link">Train</router-link>
    <router-link :to="isExp ? '/exp/predict' : '/predict'" class="nav-link">Predict</router-link>
    <span class="spacer"></span>
    <router-link v-if="!isExp" to="/exp/predict" class="nav-link toggle">→ /exp</router-link>
    <router-link v-else to="/predict" class="nav-link toggle">← /production</router-link>
  </nav>
  <router-view />
</template>

<style>
.app-nav {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 16px;
  background: #f3f4f6;
  border-bottom: 1px solid #e5e7eb;
}
.app-nav.exp {
  background: #fef3c7;
  border-bottom-color: #fcd34d;
}
.exp-tag {
  font-size: 0.85em;
  color: #92400e;
  font-weight: 600;
  margin-right: 4px;
}
.nav-link {
  text-decoration: none;
  color: #1f2937;
  padding: 4px 8px;
  border-radius: 4px;
}
.nav-link:hover {
  background: rgba(0, 0, 0, 0.05);
}
.nav-link.router-link-active {
  background: rgba(59, 130, 246, 0.15);
  color: #1d4ed8;
  font-weight: 600;
}
.nav-link.toggle {
  font-size: 0.85em;
  opacity: 0.75;
}
.spacer {
  flex: 1;
}
</style>
