<template>
  <div class="exp-page">
    <div class="exp-banner">
      🧪 <strong>Experimental</strong> — Two-Stage Sequence Recognizer (rolling-window GRU)
    </div>

    <h1>Predict (Sequence)</h1>

    <!-- 1) Model selection -->
    <div class="card">
      <label class="block-label">เลือกโมเดล (exp namespace):</label>
      <select
        v-model="selectedModel"
        @change="loadModel"
        class="version-select"
      >
        <option disabled value="">-- กรุณาเลือก --</option>
        <optgroup v-if="groupedVersions.active.length" label="🟢 Active (กำลังใช้งาน)">
          <option v-for="ver in groupedVersions.active" :key="ver.version" :value="ver.version">
            {{ formatVersionLabel(ver) }}
          </option>
        </optgroup>
        <optgroup v-if="groupedVersions.rollback.length" label="🟡 Rollback (สำรอง)">
          <option v-for="ver in groupedVersions.rollback" :key="ver.version" :value="ver.version">
            {{ formatVersionLabel(ver) }}
          </option>
        </optgroup>
        <optgroup v-if="groupedVersions.dev.length" label="⚪ Dev (ระหว่างพัฒนา)">
          <option v-for="ver in groupedVersions.dev" :key="ver.version" :value="ver.version">
            {{ formatVersionLabel(ver) }}
          </option>
        </optgroup>
      </select>
      <div v-if="!availableVersions.length" class="warn">
        ยังไม่มีโมเดล exp ใน <code>/api/exp/models</code> — ต้อง deploy GRU ก่อน
      </div>
      <div v-if="modelInfo" class="info-box">
        <div>Architecture: <code>{{ modelInfo.architecture || '?' }}</code></div>
        <div>Sequence length: {{ modelInfo.sequenceLength || '?' }} frames</div>
        <div v-if="modelInfo.valAccuracy != null">
          Validation Acc: {{ (modelInfo.valAccuracy * 100).toFixed(2) }}%
        </div>
      </div>
    </div>

    <!-- 2) Camera + skeleton overlay -->
    <div class="card">
      <div class="webcam-stack" v-if="cameraOn">
        <video
          ref="videoRef"
          autoplay
          playsinline
          muted
          class="webcam-video mirror"
        ></video>
        <canvas
          ref="overlayCanvasRef"
          class="webcam-overlay mirror"
        ></canvas>
      </div>
      <button v-if="!cameraOn" @click="startCamera" class="btn-primary">
        📷 เปิดกล้อง
      </button>
      <button v-else @click="stopCamera" class="btn-secondary">
        ⏹ หยุดกล้อง
      </button>
    </div>

    <!-- 3) Prediction display -->
    <div class="card" v-if="cameraOn">
      <div class="pred-line">
        คาดการณ์: <strong>{{ prediction || '(รอ...)' }}</strong>
      </div>
      <div class="meter">
        Window: {{ frameBuffer.length }}/{{ SEQ_LEN }}
        <span v-if="frameBuffer.length < SEQ_LEN">(กำลังเก็บเฟรม…)</span>
      </div>
      <ul class="top-list">
        <li v-for="({ prob, label }, i) in topN" :key="i">
          {{ label }} — {{ (prob * 100).toFixed(1) }}%
        </li>
      </ul>
    </div>

    <p class="note">
      🔗 ใช้ API <code>/api/exp/models</code> — production /predict ไม่กระทบ
    </p>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount, markRaw, nextTick } from 'vue'
import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-webgl'
import {
  getHandsInstance,
  setOnResults,
  normalizeLandmarks,
  drawHandSkeleton,
  clearOverlay,
} from '../../composables/useHandLandmarks.js'

const SEQ_LEN = 15
const TOP_N = 3

const videoRef = ref(null)
const overlayCanvasRef = ref(null)
const cameraOn = ref(false)
const availableVersions = ref([])
const selectedModel = ref('')
const modelInfo = ref(null)
const prediction = ref('')
const probabilities = ref([])
const frameBuffer = ref([])
const labels = ref([])

let model = null
let stream = null
let rafId = null

const topN = computed(() =>
  probabilities.value
    .map((p, i) => ({ prob: p, label: labels.value[i] || `?${i}` }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, TOP_N)
)

const groupedVersions = computed(() => {
  const out = { active: [], rollback: [], dev: [] }
  for (const v of availableVersions.value) {
    const g = v.group || 'dev'
    if (out[g]) out[g].push(v)
    else out.dev.push(v)
  }
  return out
})

function formatVersionLabel(ver) {
  const parts = [ver.version]
  if (ver.valAccuracy != null) parts.push(`${(ver.valAccuracy * 100).toFixed(1)}% val`)
  if (ver.labelsCount != null) parts.push(`${ver.labelsCount} class`)
  if (ver.sequenceLength) parts.push(`seq=${ver.sequenceLength}`)
  return parts.join(' — ')
}

async function loadAvailableVersions() {
  try {
    const res = await fetch('/api/exp/models')
    if (!res.ok) throw new Error('exp models endpoint failed')
    availableVersions.value = await res.json()
    if (availableVersions.value.length > 0) {
      selectedModel.value = availableVersions.value[0].version
      await loadModel()
    }
  } catch (e) {
    console.error('loadAvailableVersions:', e)
  }
}

async function loadModel() {
  if (!selectedModel.value) return
  try {
    if (!tf.getBackend()) {
      await tf.setBackend('webgl')
      await tf.ready()
    }
    const path = `/models/${selectedModel.value}/model.json`
    const m = await tf.loadLayersModel(path)
    model = markRaw(m)
    modelInfo.value = availableVersions.value.find(
      (v) => v.version === selectedModel.value
    )
    labels.value = modelInfo.value?.labels || []
    tf.tidy(() => {
      const dim = modelInfo.value?.inputDim || 63
      const sl = modelInfo.value?.sequenceLength || SEQ_LEN
      m.predict(tf.zeros([1, sl, dim]))
    })
    console.log('Exp model loaded:', selectedModel.value)
  } catch (e) {
    console.error('loadModel failed:', e)
  }
}

function syncOverlaySize() {
  const v = videoRef.value
  const c = overlayCanvasRef.value
  if (!v || !c) return
  if (v.videoWidth) {
    c.width = v.videoWidth
    c.height = v.videoHeight
  }
}

function onResults(results) {
  const raw = results.multiHandLandmarks?.[0] || null

  if (overlayCanvasRef.value) {
    syncOverlaySize()
    const ctx = overlayCanvasRef.value.getContext('2d')
    clearOverlay(ctx)
    if (raw) {
      drawHandSkeleton(ctx, raw, {
        width: overlayCanvasRef.value.width,
        height: overlayCanvasRef.value.height,
      })
    }
  }

  if (!raw) return

  const norm = normalizeLandmarks(raw)
  if (!norm) return

  frameBuffer.value.push(norm)
  if (frameBuffer.value.length > SEQ_LEN) {
    frameBuffer.value.shift()
  }

  if (frameBuffer.value.length < SEQ_LEN || !model) return

  const seqLen = modelInfo.value?.sequenceLength || SEQ_LEN
  const dim = modelInfo.value?.inputDim || norm.length
  const window = frameBuffer.value.slice(-seqLen)
  const flat = window.flat()
  const logits = tf.tidy(() => {
    const x = tf.tensor3d(flat, [1, seqLen, dim])
    const out = model.predict(x)
    const data = out.dataSync()
    probabilities.value = [...data]
    return data
  })
  const idx = logits.indexOf(Math.max(...logits))
  prediction.value = labels.value[idx] || '(?)'
}

async function startCamera() {
  cameraOn.value = true
  await nextTick()
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user' },
      audio: false,
    })
    videoRef.value.srcObject = stream
    await videoRef.value.play()
    videoRef.value.addEventListener('loadedmetadata', syncOverlaySize, { once: true })
  } catch (e) {
    console.error('Camera error:', e)
    cameraOn.value = false
    return
  }

  setOnResults(onResults)
  const hands = getHandsInstance()

  const loop = async () => {
    if (videoRef.value && videoRef.value.readyState >= 2) {
      await hands.send({ image: videoRef.value })
    }
    if (cameraOn.value) {
      rafId = requestAnimationFrame(loop)
    }
  }
  rafId = requestAnimationFrame(loop)
}

function stopCamera() {
  if (rafId) {
    cancelAnimationFrame(rafId)
    rafId = null
  }
  if (stream) {
    stream.getTracks().forEach((t) => t.stop())
    stream = null
  }
  if (videoRef.value) videoRef.value.srcObject = null
  cameraOn.value = false
  frameBuffer.value = []
  prediction.value = ''
  probabilities.value = []
}

onMounted(async () => {
  await loadAvailableVersions()
})

onBeforeUnmount(() => {
  stopCamera()
  if (model) model.dispose()
})
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
h1 { font-size: 1.4rem; margin: 0 0 12px; }
.card {
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 6px;
  padding: 10px 12px;
  margin-bottom: 12px;
}
.block-label { display: block; font-weight: 600; margin-bottom: 6px; }
.version-select {
  padding: 6px 8px;
  border: 1px solid #888;
  border-radius: 4px;
  min-width: 280px;
  max-width: 100%;
}
.info-box {
  margin-top: 8px;
  padding: 8px;
  background: rgba(0, 100, 200, 0.1);
  border-left: 3px solid #4a90e2;
  border-radius: 4px;
  font-size: 0.9em;
}
.warn { color: #d97706; margin-top: 6px; font-size: 0.9em; }
.btn-primary, .btn-secondary {
  padding: 8px 16px;
  border-radius: 4px;
  border: 1px solid #888;
  cursor: pointer;
  background: #2563eb;
  color: #fff;
}
.btn-secondary { background: #6b7280; }
.pred-line { font-size: 1.1em; }
.meter { font-size: 0.9em; color: #6b7280; margin: 6px 0; }
.top-list { padding-left: 20px; }
.note { font-size: 0.8em; color: #6b7280; }

.mirror { transform: scaleX(-1); }
.webcam-stack {
  position: relative;
  display: inline-block;
  line-height: 0;
  border: 1px solid #555;
  border-radius: 6px;
  overflow: hidden;
  margin-bottom: 10px;
}
.webcam-video { display: block; width: 320px; height: auto; }
.webcam-overlay {
  position: absolute;
  top: 0; left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}
</style>
