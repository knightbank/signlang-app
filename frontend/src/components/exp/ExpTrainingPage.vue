<template>
  <div class="exp-page">
    <div class="exp-banner">
      🧪 <strong>Experimental</strong> — Sequence-based training (clip recording, not single-frame)
    </div>
    <h1>Train (Sequence)</h1>
    <div class="placeholder">
      <p>หน้านี้คือพื้นที่ทดลอง <code>/exp/train</code> สำหรับเก็บคลิป training (1-2 วินาที) ไม่ใช่ภาพเดียว</p>
      <p>API namespace ที่หน้านี้จะใช้:</p>
      <ul>
        <li><code>GET /api/exp/models</code> — base model สำหรับ fine-tune</li>
        <li><code>POST /api/exp/save-model</code> — บันทึก GRU + ลำดับ landmark</li>
        <li><code>GET /api/exp/dataset/&lt;scope&gt;/&lt;name&gt;</code> — replay buffer</li>
        <li><code>POST /api/exp/promote-model</code> — promote เฉพาะภายใน exp/</li>
      </ul>
      <p>จะรองรับ workflow expert review (playback + confirm) ในอนาคต</p>
      <p>หน้า training เดิมยังใช้ได้: <router-link to="/train">/train (production)</router-link></p>
    </div>
  </div>
</template>

<script setup>
</script>

<style scoped>
.exp-page {
  padding: 16px;
  max-width: 900px;
  margin: 0 auto;
  font-family: system-ui, sans-serif;
}
.exp-banner {
  background: #fef3c7;
  color: #92400e;
  border: 1px solid #fcd34d;
  border-radius: 6px;
  padding: 8px 12px;
  margin-bottom: 12px;
  font-size: 0.9em;
}
h1 {
  font-size: 1.4rem;
  margin: 0 0 12px;
}
.placeholder {
  background: rgba(255, 255, 255, 0.04);
  border: 1px dashed rgba(255, 255, 255, 0.2);
  padding: 16px;
  border-radius: 6px;
}
.placeholder p {
  margin: 6px 0;
}
</style>
