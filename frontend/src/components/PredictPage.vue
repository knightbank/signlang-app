<template>
  <div class="p-4 space-y-4">
    <h1 class="text-xl font-bold">Predict Sign Language</h1>

    <!-- 1) Model Selection -->
    <div>
      <label class="block font-medium mb-1">เลือกโมเดล:</label>
      <select
        v-model="selectedModel"
        @change="loadModel"
        class="p-2 border rounded w-full md:w-1/2"
      >
        <option disabled value="">-- กรุณาเลือก Model --</option>
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
    </div>

    <!-- 2) Loaded model info + tooltip -->
    <div v-if="modelName" class="flex items-center space-x-2 text-sm text-gray-600">
      <span>📦 ใช้งานโมเดล: <code>{{ modelName }}</code></span>
      <button
        @click="toggleTooltip"
        class="relative text-blue-600 hover:text-blue-800 focus:outline-none"
        aria-label="Show model metrics"
      >
        ℹ️
        <div
          v-if="showTooltip"
          class="absolute top-6 left-0 w-56 p-2 bg-gray-800 text-white text-xs rounded shadow-lg z-10 text-left"
        >
          <div><strong>Metrics:</strong></div>
          <div>Parameters: {{ parameterCount.toLocaleString() }}</div>
          <div v-if="loadedMetrics.accuracy !== null">
            Validation Accuracy: {{ (loadedMetrics.accuracy * 100).toFixed(2) }}%
          </div>
          <div v-if="loadedMetrics.loss !== null">
            Validation Loss: {{ loadedMetrics.loss.toFixed(4) }}
          </div>
          <div v-if="loadedMetrics.datasetSize">
            Training samples: {{ loadedMetrics.datasetSize }}
          </div>
          <div v-if="!hasAnyMetric" class="text-gray-300 italic">ไม่มีข้อมูล metric</div>
        </div>
      </button>
    </div>

    <!-- 3) Mode -->
    <div>
      <label class="block font-medium mb-1">เลือกโหมด:</label>
      <select v-model="mode" class="p-2 border rounded w-full md:w-1/2">
        <option value="image">Predict จากรูปภาพ</option>
        <option value="webcam">Predict จากกล้องแบบเรียลไทม์</option>
      </select>
    </div>

    <!-- 4) Image mode -->
    <div v-if="mode === 'image'">
      <label class="block mb-1 font-medium">เลือกรูปภาพ:</label>
      <input
        type="file"
        accept="image/*"
        @change="handleImageUpload"
        class="mb-2"
      />
      <div v-if="imageDataUrl" class="flex justify-center">
        <canvas
          ref="imageCanvasRef"
          class="border rounded max-w-full h-auto"
        ></canvas>
      </div>
      <button
        v-if="imageDataUrl"
        @click="predictFromImage"
        class="mt-2 px-4 py-2 bg-blue-600 text-white rounded"
      >
        🔍 Predict จากภาพ
      </button>
      <div
        v-if="prediction"
        class="mt-4 p-3 bg-green-100 border rounded text-center text-lg"
      >
        ผลลัพธ์ที่คาดการณ์: <strong>{{ prediction }}</strong>
        <ul class="list-disc ml-6 mt-2 text-left">
          <li
            v-for="({ prob, label }, index) in topNPredictions"
            :key="index"
          >
            {{ label }} — {{ (prob * 100).toFixed(2) }}%
          </li>
        </ul>
      </div>
    </div>

    <!-- 5) Webcam mode -->
    <div v-if="mode === 'webcam'" class="flex flex-col items-center space-y-4">
      <div class="w-full bg-green-100 border rounded p-3">
        <div class="text-center text-lg">
          ผลลัพธ์ที่คาดการณ์:
          <strong v-if="prediction">{{ prediction }}</strong>
          <strong v-else class="text-gray-700">(รอก่อน…)</strong>
        </div>
        <ul class="list-disc ml-6 mt-2">
          <li
            v-for="({ prob, label }, index) in topNPredictions"
            :key="index"
          >
            {{ label }} — {{ (prob * 100).toFixed(2) }}%
          </li>
          <li v-if="!probabilities.length" class="text-gray-500">
            กำลังตรวจจับมือ…
          </li>
        </ul>
      </div>

      <div class="webcam-stack">
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
    </div>
  </div>
</template>

<script setup>
import {
  ref,
  watch,
  onMounted,
  onBeforeUnmount,
  markRaw,
  computed,
  nextTick,
} from 'vue'
import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-webgl'
import {
  getHandsInstance,
  setOnResults,
  handsSend,
  handsSendStatic,
  normalizeLandmarks,
  drawHandSkeleton,
  clearOverlay,
} from '../composables/useHandLandmarks.js'

const videoRef = ref(null)
const overlayCanvasRef = ref(null)
const imageCanvasRef = ref(null)
const imageDataUrl = ref(null)
const model = ref(null)
const labels = ref([])
const prediction = ref('')
const probabilities = ref([])
const mode = ref('image')
const modelName = ref('')
const selectedModel = ref('')
const availableVersions = ref([])

const showTooltip = ref(false)
const parameterCount = ref(0)
const loadedMetrics = ref({ accuracy: null, loss: null, datasetSize: null, normalized: null })
const hasAnyMetric = computed(
  () => loadedMetrics.value.accuracy !== null || loadedMetrics.value.loss !== null
)
const useNormalized = computed(() => loadedMetrics.value.normalized === true)

const topNPredictions = computed(() =>
  probabilities.value
    .map((prob, i) => ({ prob, label: labels.value[i] }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 3)
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

let animationFrameId = null
let webcamLandmarks = null

async function initializeTfjsBackend() {
  await tf.setBackend('webgl')
  await tf.ready()
}

function formatVersionLabel(ver) {
  if (typeof ver === 'string') return ver
  const parts = [ver.version]
  if (ver.accuracy != null) parts.push(`${(ver.accuracy * 100).toFixed(1)}% acc`)
  if (ver.labelsCount != null) parts.push(`${ver.labelsCount} class`)
  if (ver.datasetSize) parts.push(`${ver.datasetSize} samples`)
  return parts.join(' — ')
}

async function loadAvailableVersions() {
  try {
    const res = await fetch('/api/models')
    if (!res.ok) throw new Error('ไม่สามารถดึงรายการโมเดลได้')
    const versions = await res.json()
    availableVersions.value = versions.map((v) =>
      typeof v === 'string' ? { version: v } : v
    )
    if (availableVersions.value.length > 0) {
      selectedModel.value = availableVersions.value[0].version
      await loadModel()
    }
  } catch (e) {
    console.error('loadAvailableVersions error:', e)
    availableVersions.value = []
  }
}

async function loadModel() {
  if (!selectedModel.value) return
  try {
    if (!tf.getBackend()) {
      await initializeTfjsBackend()
    }

    modelName.value = selectedModel.value
    const modelPath = `/models/${selectedModel.value}/model.json`
    const loadedModel = await tf.loadLayersModel(modelPath)
    model.value = markRaw(loadedModel)
    parameterCount.value = loadedModel.countParams()

    const labelsPath = `/models/${selectedModel.value}/labels.json`
    const resLabels = await fetch(labelsPath)
    if (!resLabels.ok) throw new Error('โหลด labels.json ไม่สำเร็จ')
    labels.value = await resLabels.json()

    try {
      const resMet = await fetch(`/models/${selectedModel.value}/metrics.json`)
      if (resMet.ok) {
        const data = await resMet.json()
        loadedMetrics.value.accuracy = data.accuracy ?? null
        loadedMetrics.value.loss = data.loss ?? null
        loadedMetrics.value.datasetSize = data.datasetSize ?? null
        loadedMetrics.value.normalized = data.normalized === true
      } else {
        loadedMetrics.value = { accuracy: null, loss: null, datasetSize: null, normalized: null }
      }
    } catch {
      loadedMetrics.value = { accuracy: null, loss: null, datasetSize: null, normalized: null }
    }

    const inputDim = loadedModel.inputs[0].shape[1] || 63
    tf.tidy(() => loadedModel.predict(tf.zeros([1, inputDim])))
    console.log('✅ โมเดล, labels และ metrics ถูกโหลดแล้ว')
  } catch (err) {
    console.error('โหลดโมเดลไม่สำเร็จ:', err)
    modelName.value = ''
    parameterCount.value = 0
    loadedMetrics.value = { accuracy: null, loss: null, datasetSize: null }
  }
}

function toggleTooltip() {
  showTooltip.value = !showTooltip.value
}

function handleImageUpload(e) {
  const f = e.target.files[0]
  if (f) {
    imageDataUrl.value = URL.createObjectURL(f)
    prediction.value = ''
    probabilities.value = []
    nextTick(() => drawImageOnCanvas())
  }
}

async function drawImageOnCanvas() {
  if (!imageCanvasRef.value || !imageDataUrl.value) return
  const img = new Image()
  img.src = imageDataUrl.value
  await new Promise((r) => (img.onload = r))
  const canvas = imageCanvasRef.value
  canvas.width = img.width
  canvas.height = img.height
  const ctx = canvas.getContext('2d')
  ctx.drawImage(img, 0, 0)
}

async function predictFromImage() {
  prediction.value = '...กำลังประมวลผล...'
  probabilities.value = []
  try {
    const canvas = imageCanvasRef.value
    if (!canvas) return
    await drawImageOnCanvas()

    const results = await handsSendStatic(canvas)
    const raw = results.multiHandLandmarks?.[0] || null
    if (!raw) {
      prediction.value = 'ไม่พบมือ'
      return
    }

    const ctx = canvas.getContext('2d')
    drawHandSkeleton(ctx, raw, {
      width: canvas.width,
      height: canvas.height,
      lineWidth: Math.max(3, canvas.width / 200),
      dotRadius: Math.max(4, canvas.width / 150),
      wristRadius: Math.max(8, canvas.width / 80),
    })

    const input = useNormalized.value
      ? normalizeLandmarks(raw)
      : raw.flatMap((p) => [p.x, p.y, p.z])
    if (!input) {
      prediction.value = 'ตรวจจับ landmark ผิดพลาด'
      return
    }

    const logits = tf.tidy(() => {
      const out = model.value.predict(
        tf.tensor2d(input, [1, input.length], 'float32')
      )
      const data = out.dataSync()
      probabilities.value = [...data]
      return data
    })

    const idx = logits.indexOf(Math.max(...logits))
    prediction.value = labels.value[idx] || '(ไม่รู้จัก)'
  } catch (e) {
    console.error('Error predict image:', e)
    prediction.value = '❌เกิดข้อผิดพลาด'
  }
}

function stopCamera() {
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId)
    animationFrameId = null
  }
  if (videoRef.value && videoRef.value.srcObject) {
    videoRef.value.srcObject.getTracks().forEach((t) => t.stop())
    videoRef.value.srcObject = null
  }
  webcamLandmarks = null
}

function syncOverlaySize() {
  if (!overlayCanvasRef.value || !videoRef.value) return
  const v = videoRef.value
  const c = overlayCanvasRef.value
  if (v.videoWidth && v.videoHeight) {
    c.width = v.videoWidth
    c.height = v.videoHeight
  }
}

function onWebcamResults(results) {
  webcamLandmarks = results.multiHandLandmarks?.[0] || null

  if (!webcamLandmarks) {
    prediction.value = 'ไม่พบมือ'
    probabilities.value = []
    if (overlayCanvasRef.value) {
      const ctx = overlayCanvasRef.value.getContext('2d')
      clearOverlay(ctx)
    }
    return
  }

  if (overlayCanvasRef.value) {
    const ctx = overlayCanvasRef.value.getContext('2d')
    syncOverlaySize()
    clearOverlay(ctx)
    drawHandSkeleton(ctx, webcamLandmarks, {
      width: overlayCanvasRef.value.width,
      height: overlayCanvasRef.value.height,
    })
  }

  if (!model.value) return

  const input = useNormalized.value
    ? normalizeLandmarks(webcamLandmarks)
    : webcamLandmarks.flatMap((p) => [p.x, p.y, p.z])
  if (!input) return

  const logits = tf.tidy(() => {
    const out = model.value.predict(
      tf.tensor2d(input, [1, input.length], 'float32')
    )
    const data = out.dataSync()
    probabilities.value = [...data]
    return data
  })

  const i = logits.indexOf(Math.max(...logits))
  prediction.value = labels.value[i]
}

watch(mode, async (newMode) => {
  stopCamera()

  if (newMode === 'webcam') {
    await nextTick()

    if (!model.value) {
      try {
        await loadModel()
      } catch (e) {
        console.error('โหลดโมเดลก่อนเปิดกล้องล้มเหลว:', e)
        return
      }
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user' },
        audio: false,
      })
      videoRef.value.srcObject = stream
      await videoRef.value.play()
      videoRef.value.addEventListener('loadedmetadata', syncOverlaySize, { once: true })
    } catch (err) {
      console.error('Error accessing camera:', err)
      return
    }

    setOnResults(onWebcamResults)
    const hands = getHandsInstance()

    const loop = async () => {
      if (videoRef.value && videoRef.value.readyState >= 2) {
        await hands.send({ image: videoRef.value })
      }
      animationFrameId = requestAnimationFrame(loop)
    }
    animationFrameId = requestAnimationFrame(loop)
  } else {
    prediction.value = ''
    probabilities.value = []
  }
})

onMounted(async () => {
  await loadAvailableVersions()
})

onBeforeUnmount(() => {
  stopCamera()
  if (model.value) model.value.dispose()
  tf.disposeVariables()
})
</script>

<style scoped>
video,
img {
  max-width: 100%;
  height: auto;
}
.mirror {
  transform: scaleX(-1);
}
.webcam-stack {
  position: relative;
  display: inline-block;
  line-height: 0;
  border: 1px solid #ccc;
  border-radius: 6px;
  overflow: hidden;
}
.webcam-video {
  display: block;
  width: 320px;
  height: auto;
}
.webcam-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}
</style>
