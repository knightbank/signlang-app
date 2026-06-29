import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    host: true,       // ฟังทุก interface (0.0.0.0)
    //https: true,      // ใช้ HTTPS
    port: 5173,       // พอร์ตตามที่คุณใช้ (ค่าดีฟอลต์คือ 5173)
    strictPort: true, // ถ้าพอร์ตไม่ว่างให้ error ทันที
    proxy: {
      '/api': 'http://localhost:3001',
      '/models': 'http://localhost:3001'
    }
  }
})
