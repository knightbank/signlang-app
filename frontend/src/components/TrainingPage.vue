<template>
  <div class="train-page">
    <h1>Sign Language Recognition – Training</h1>

    <!-- 1) Training mode -->
    <div class="card">
      <strong>โหมดการเทรน:</strong>
      <label class="radio">
        <input type="radio" v-model="trainingMode" value="new" :disabled="isTraining" />
        🆕 สร้างโมเดลใหม่ (จากศูนย์)
      </label>
      <label class="radio">
        <input type="radio" v-model="trainingMode" value="finetune" :disabled="isTraining" />
        🔧 ต่อยอดโมเดลเดิม (สอนสะสม)
      </label>
    </div>

    <!-- 2) Base model selection (finetune mode) -->
    <div v-if="trainingMode === 'finetune'" class="card">
      <label class="block-label">เลือกโมเดลพื้นฐาน:</label>
      <select
        v-model="selectedVersion"
        @change="loadBaseModelAndBuffer"
        :disabled="isTraining"
        class="version-select"
      >
        <option disabled value="">-- กรุณาเลือกเวอร์ชัน --</option>
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

      <div v-if="baseModelInfo" class="info-box">
        <div>📚 <strong>ข้อมูลสะสมในโมเดลนี้:</strong></div>
        <div>คลาส: {{ baseModelInfo.labels.join(', ') }}</div>
        <div>จำนวนตัวอย่างเดิม: {{ baseDataset.length.toLocaleString() }} samples</div>
        <div v-if="baseModelInfo.accuracy != null">
          ความแม่นยำเดิม: {{ (baseModelInfo.accuracy * 100).toFixed(2) }}%
        </div>
        <div v-if="!baseModelInfo.normalized" class="warn">
          ⚠️ โมเดลนี้ใช้ feature แบบเดิม (ไม่ normalize) — ต่อยอดจะรีเซ็ตเป็นแบบใหม่
        </div>
      </div>
    </div>

    <!-- 3) 44-letter picker grid -->
    <div class="card">
      <strong>เลือกพยัญชนะที่จะสอน:</strong>
      <div class="picker-grid">
        <button
          v-for="ch in THAI_CONSONANTS"
          :key="ch"
          @click="toggleClassByLetter(ch)"
          :class="['picker-btn', letterState(ch)]"
          :disabled="isTraining"
          :title="THAI_CONSONANT_NAMES[ch] || ch"
        >
          <span class="ch">{{ ch }}</span>
          <span v-if="getClass(ch)?.landmarkCount" class="badge">
            {{ getClass(ch).landmarkCount }}
          </span>
        </button>
      </div>
      <button
        @click="addCustomClass"
        class="btn-secondary"
        :disabled="isTraining"
      >
        + เพิ่มคลาสชื่ออื่น (ไม่ใช่พยัญชนะ)
      </button>
    </div>

    <!-- 4) Per-class blocks -->
    <div v-if="classes.length" class="class-grid">
      <div
        v-for="(cls, index) in classes"
        :key="cls.id"
        class="class-block"
      >
        <div class="class-head">
          <input
            v-model="cls.name"
            :placeholder="`ชื่อคลาส ${index + 1}`"
            class="class-name"
            @focus="$event.target.select()"
          />
          <button
            @click="removeClass(index)"
            class="x-btn"
            :disabled="isTraining"
            title="ลบคลาสนี้"
          >×</button>
        </div>

        <div class="counter">
          ตรวจพบมือ:
          <strong>{{ cls.landmarkCount }}/{{ cls.attemptCount }}</strong>
          <span v-if="cls.landmarkCount < 30" class="warn-text">
            (แนะนำ ≥ 30)
          </span>
        </div>
        <div v-if="cls.processing || cls.processingLabel" class="proc-row">
          <span v-if="cls.processing" class="spin">⏳</span>
          <span>{{ cls.processingLabel }}</span>
        </div>

        <div class="input-row">
          <label class="file-label">
            🎥 วิดีโอ:
            <input
              type="file"
              accept="video/*"
              @change="e => handleUploadVideo(e, index)"
              :disabled="isTraining || cls.processing"
            />
          </label>
          <span v-if="cls.videoName" class="filename">{{ cls.videoName }}</span>
        </div>

        <div class="input-row">
          <label class="file-label">
            🖼️ รูปภาพ (หลายไฟล์):
            <input
              type="file"
              accept="image/*"
              multiple
              @change="e => handleUploadImages(e, index)"
              :disabled="isTraining || cls.processing"
            />
          </label>
          <button
            @click="openCamera(index)"
            class="btn-camera"
            :disabled="isTraining"
          >
            📷 ถ่ายจากกล้อง
          </button>
        </div>

        <div v-if="cls.thumbs.length" class="thumb-strip">
          <div
            v-for="(t, ti) in cls.thumbs"
            :key="ti"
            :class="['thumb', t.detected ? 'detected' : 'not-detected']"
            :title="t.detected ? 'ตรวจพบมือ (แตะเพื่อขยาย)' : 'ไม่พบมือ (แตะเพื่อขยาย)'"
            @click="openThumb(t)"
          >
            <img :src="t.dataUrl" alt="thumb" />
            <span v-if="!t.detected" class="thumb-tag">ไม่พบ</span>
          </div>
        </div>
      </div>
    </div>

    <!-- 5) Train button + augmentation -->
    <div class="card">
      <label class="checkbox">
        <input type="checkbox" v-model="useAugmentation" :disabled="isTraining" />
        เพิ่มข้อมูลด้วย augmentation (jitter + mirror flip)
      </label>
      <button
        @click="trainModel"
        :disabled="isTraining || !classes.length"
        class="btn-train"
      >
        🧠 เทรนโมเดล
      </button>
    </div>

    <!-- 6) Progress -->
    <pre v-if="progressText" class="progress">{{ progressText }}</pre>

    <!-- 7) Delta result -->
    <div v-if="lastResult" class="card result-box">
      <div><strong>ผลการเทรน:</strong></div>
      <div>Validation Accuracy: {{ (lastResult.valAccuracy * 100).toFixed(2) }}%</div>
      <div v-if="lastResult.baselineAccuracy != null">
        ความแม่นเดิม: {{ (lastResult.baselineAccuracy * 100).toFixed(2) }}%
      </div>
      <div v-if="lastResult.delta != null" :class="lastResult.delta >= 0 ? 'good' : 'bad'">
        {{ lastResult.delta >= 0 ? '📈 ดีขึ้น' : '📉 แย่ลง' }}
        {{ (Math.abs(lastResult.delta) * 100).toFixed(2) }}%
      </div>
      <div>บันทึกเป็น: <code>{{ lastResult.path }}</code></div>
      <div>ตัวอย่างทั้งหมด (สะสม): {{ lastResult.datasetSize }}</div>
    </div>

    <!-- Thumbnail viewer modal -->
    <div v-if="viewerThumb" class="modal-bg" @click.self="closeThumb">
      <div class="thumb-viewer">
        <div class="thumb-viewer-head">
          <strong>{{ viewerThumb.detected ? '✅ ตรวจพบมือ' : '❌ ไม่พบมือ' }}</strong>
          <button @click="closeThumb" class="x-btn">×</button>
        </div>
        <img :src="viewerThumb.dataUrl" alt="full" class="thumb-viewer-img" />
      </div>
    </div>

    <!-- Camera capture modal -->
    <div v-if="cameraOpen" class="modal-bg" @click.self="closeCamera">
      <div class="modal">
        <div class="modal-head">
          <strong>📷 เก็บตัวอย่างจากกล้อง — คลาส "{{ classes[cameraIndex]?.name }}"</strong>
          <button @click="closeCamera" class="x-btn">×</button>
        </div>
        <div class="webcam-stack">
          <video
            ref="captureVideoRef"
            autoplay
            playsinline
            muted
            class="webcam-video mirror"
          ></video>
          <canvas
            ref="captureOverlayRef"
            class="webcam-overlay mirror"
          ></canvas>
        </div>
        <div class="capture-controls">
          <div class="capture-status">
            เก็บได้: <strong>{{ captureCollected }}</strong> ตัวอย่าง
            {{ captureBurstActive ? '(กำลังเก็บอัตโนมัติ…)' : '' }}
          </div>
          <button
            @click="captureSingle"
            :disabled="captureBurstActive"
            class="btn-capture"
          >
            📸 บันทึก 1 ภาพ
          </button>
          <button
            v-if="!captureBurstActive"
            @click="captureBurst(60)"
            class="btn-capture"
          >
            📸 บันทึก 60 ภาพ (auto, 200 ms)
          </button>
          <button
            v-else
            @click="stopBurst"
            class="btn-capture-stop"
          >
            ⏹ หยุดเก็บ
          </button>
          <button @click="closeCamera" class="btn-secondary">ปิดกล้อง</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount, nextTick } from 'vue'
import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-webgl'
import { THAI_CONSONANTS, THAI_CONSONANT_NAMES } from '../constants/thaiConsonants.js'
import {
  getHandsInstance,
  setOnResults,
  handsSend,
  handsSendStatic,
  normalizeLandmarks,
  drawHandSkeleton,
  clearOverlay,
} from '../composables/useHandLandmarks.js'

const trainingMode = ref('new')
const selectedVersion = ref('')
const availableVersions = ref([])
const baseModelInfo = ref(null)
const baseDataset = ref([])
let baseModel = null

const groupedVersions = computed(() => {
  const out = { active: [], rollback: [], dev: [] }
  for (const v of availableVersions.value) {
    const g = v.group || 'dev'
    if (out[g]) out[g].push(v)
    else out.dev.push(v)
  }
  return out
})

const classes = ref([])
const progressText = ref('')
const isTraining = ref(false)
const lastResult = ref(null)
const useAugmentation = ref(true)

let classIdSeq = 0
function makeClass(name) {
  return {
    id: ++classIdSeq,
    name,
    videoName: '',
    videoFile: null,
    imageFiles: [],
    capturedLandmarks: [],
    // Pre-extracted samples from uploaded files (so we don't re-extract at train time)
    uploadedSamples: [],
    landmarkCount: 0,
    attemptCount: 0,
    thumbs: [],
    processing: false,
    processingLabel: '',
  }
}

function getClass(name) {
  return classes.value.find((c) => c.name === name)
}

function letterState(ch) {
  const c = getClass(ch)
  if (!c) return 'inactive'
  if (c.landmarkCount > 0) return 'has-data'
  return 'active'
}

function toggleClassByLetter(ch) {
  const idx = classes.value.findIndex((c) => c.name === ch)
  if (idx >= 0) {
    const el = document.querySelector(`[data-class-id="${classes.value[idx].id}"]`)
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' })
    return
  }
  classes.value.push(makeClass(ch))
}

function addCustomClass() {
  classes.value.push(makeClass(`Class ${classes.value.length + 1}`))
}

function removeClass(index) {
  classes.value.splice(index, 1)
}

function formatVersionLabel(ver) {
  const parts = [ver.version]
  if (ver.accuracy != null) parts.push(`${(ver.accuracy * 100).toFixed(1)}% acc`)
  if (ver.labelsCount != null) parts.push(`${ver.labelsCount} class`)
  if (ver.datasetSize) parts.push(`${ver.datasetSize} samples`)
  return parts.join(' — ')
}

async function loadAvailableVersions() {
  try {
    const res = await fetch('/api/models')
    if (!res.ok) throw new Error('โหลดรายการโมเดลไม่สำเร็จ')
    const raw = await res.json()
    availableVersions.value = raw.map((v) =>
      typeof v === 'string' ? { version: v } : v
    )
  } catch (e) {
    console.error(e)
    availableVersions.value = []
  }
}

async function loadBaseModelAndBuffer() {
  baseModel = null
  baseModelInfo.value = null
  baseDataset.value = []
  if (!selectedVersion.value) return

  const info = availableVersions.value.find((v) => v.version === selectedVersion.value)
  baseModelInfo.value = info || null

  try {
    const m = await tf.loadLayersModel(`/models/${selectedVersion.value}/model.json`)
    m.layers.forEach((l) => (l.trainable = true))
    baseModel = m
  } catch (e) {
    console.warn('โหลดโมเดลพื้นฐานไม่สำเร็จ — จะสร้างใหม่:', e)
    baseModel = null
  }

  try {
    const res = await fetch(`/models/${selectedVersion.value}/dataset.json`)
    if (res.ok) {
      const ds = await res.json()
      if (Array.isArray(ds)) baseDataset.value = ds
    }
  } catch {
    baseDataset.value = []
  }
}

async function handleUploadVideo(e, index) {
  const file = e.target.files[0]
  if (!file) return
  const cls = classes.value[index]
  cls.videoName = file.name
  cls.videoFile = file
  cls.processing = true
  cls.processingLabel = 'กำลังประมวลผลวิดีโอ...'
  try {
    const samples = await extractFromVideo(file, cls)
    for (const s of samples) cls.uploadedSamples.push(s)
    cls.processingLabel = `วิดีโอเสร็จ: เก็บได้ ${samples.length} เฟรม`
  } catch (err) {
    console.error('handleUploadVideo error:', err)
    cls.processingLabel = 'วิดีโอประมวลผลล้มเหลว'
  } finally {
    cls.processing = false
    setTimeout(() => { cls.processingLabel = '' }, 2500)
  }
}

async function handleUploadImages(e, index) {
  const files = Array.from(e.target.files)
  if (!files.length) return
  const cls = classes.value[index]
  cls.imageFiles = [...cls.imageFiles, ...files]
  cls.processing = true
  let done = 0
  for (const f of files) {
    cls.processingLabel = `กำลังประมวลผลรูป ${++done}/${files.length}`
    try {
      const r = await extractFromImage(f, cls)
      if (r.detected && r.normalized) {
        cls.uploadedSamples.push(r.normalized)
      }
    } catch (err) {
      console.error('handleUploadImages error:', err)
    }
  }
  cls.processing = false
  cls.processingLabel = `รูปเสร็จ: ${cls.landmarkCount}/${cls.attemptCount} ตรวจพบมือ`
  setTimeout(() => { cls.processingLabel = '' }, 3000)
}

async function extractFromImage(imgFile, cls) {
  const img = new Image()
  img.src = URL.createObjectURL(imgFile)
  await new Promise((res, rej) => {
    img.onload = res
    img.onerror = rej
  }).catch(() => null)

  const w = img.naturalWidth || img.width
  const h = img.naturalHeight || img.height
  if (!w || !h) {
    console.warn('extractFromImage: image has no dimensions', imgFile.name)
    cls.attemptCount++
    return { normalized: null, detected: false }
  }

  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')
  ctx.drawImage(img, 0, 0, w, h)

  cls.attemptCount++
  const results = await handsSendStatic(canvas)
  const raw = results.multiHandLandmarks?.[0] || null

  if (raw) {
    drawHandSkeleton(ctx, raw, {
      width: canvas.width,
      height: canvas.height,
      lineWidth: Math.max(2, canvas.width / 200),
    })
    cls.landmarkCount++
    pushThumb(cls, canvas, true)
    return { normalized: normalizeLandmarks(raw), detected: true }
  } else {
    pushThumb(cls, canvas, false)
    return { normalized: null, detected: false }
  }
}

async function extractFromVideo(videoFile, cls) {
  const video = document.createElement('video')
  video.src = URL.createObjectURL(videoFile)
  video.muted = true
  video.playsInline = true
  await new Promise((r) => {
    if (video.readyState >= 1) r()
    else video.onloadedmetadata = () => r()
  })

  const canvas = document.createElement('canvas')
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight
  const ctx = canvas.getContext('2d')

  // Skip transition frames at the start (raising hand) and end (lowering hand)
  // and drop any frame where the hand is still moving significantly between samples.
  const TRIM_HEAD = 0.5
  const TRIM_TAIL = 0.3
  // Average per-keypoint x/y displacement between consecutive sampled frames.
  // 0.04 ~ keypoint moved 4% of the frame width on average — clearly mid-motion.
  const STABILITY_THRESHOLD = 0.04
  const step = 0.3
  const startT = Math.min(TRIM_HEAD, video.duration / 3)
  const endT = Math.max(video.duration - TRIM_TAIL, video.duration * 2 / 3)

  const samples = []
  let rejectedMoving = 0
  let prevRaw = null
  let t = startT
  while (t < endT) {
    video.currentTime = t
    await new Promise((r) => (video.onseeked = () => r()))
    if (!video.videoWidth) {
      t += step
      continue
    }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    cls.attemptCount++
    const results = await handsSendStatic(canvas)
    const raw = results.multiHandLandmarks?.[0] || null
    if (raw) {
      // Stability check: only keep this frame if the hand has not moved
      // much since the previous accepted frame. First detected frame is
      // always accepted (no baseline yet).
      let motion = 0
      if (prevRaw) {
        for (let i = 0; i < 21; i++) {
          const dx = raw[i].x - prevRaw[i].x
          const dy = raw[i].y - prevRaw[i].y
          motion += Math.sqrt(dx * dx + dy * dy)
        }
        motion /= 21
      }
      if (!prevRaw || motion < STABILITY_THRESHOLD) {
        const norm = normalizeLandmarks(raw)
        if (norm) samples.push(norm)
        cls.landmarkCount++
        if (samples.length % 5 === 1) {
          drawHandSkeleton(ctx, raw, { width: canvas.width, height: canvas.height })
          pushThumb(cls, canvas, true)
        }
        prevRaw = raw
      } else {
        rejectedMoving++
      }
    }
    t += step
  }
  if (rejectedMoving > 0) {
    console.log(`extractFromVideo: dropped ${rejectedMoving} mid-motion frame(s) (stability threshold ${STABILITY_THRESHOLD})`)
  }
  return samples
}

function pushThumb(cls, canvas, detected) {
  if (cls.thumbs.length >= 8) cls.thumbs.shift()
  const max = Math.max(canvas.width, canvas.height)
  const ratio = max > 480 ? 480 / max : 1
  const target = document.createElement('canvas')
  target.width = Math.round(canvas.width * ratio)
  target.height = Math.round(canvas.height * ratio)
  target.getContext('2d').drawImage(canvas, 0, 0, target.width, target.height)
  cls.thumbs.push({ dataUrl: target.toDataURL('image/jpeg', 0.75), detected })
}

// —— Thumbnail viewer ——
const viewerThumb = ref(null)
function openThumb(t) {
  viewerThumb.value = t
}
function closeThumb() {
  viewerThumb.value = null
}

// —— Camera capture modal ——
const cameraOpen = ref(false)
const cameraIndex = ref(-1)
const captureVideoRef = ref(null)
const captureOverlayRef = ref(null)
const captureCollected = ref(0)
const captureBurstActive = ref(false)
let captureStream = null
let captureLoopId = null
let burstTimerId = null
let latestCaptureLandmarks = null

async function openCamera(index) {
  cameraIndex.value = index
  cameraOpen.value = true
  captureCollected.value = 0
  await nextTick()

  try {
    captureStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user' },
      audio: false,
    })
    captureVideoRef.value.srcObject = captureStream
    await captureVideoRef.value.play()
  } catch (e) {
    console.error('เปิดกล้องไม่สำเร็จ:', e)
    closeCamera()
    return
  }

  setOnResults((results) => {
    latestCaptureLandmarks = results.multiHandLandmarks?.[0] || null
    if (!captureOverlayRef.value) return
    const c = captureOverlayRef.value
    if (captureVideoRef.value.videoWidth) {
      c.width = captureVideoRef.value.videoWidth
      c.height = captureVideoRef.value.videoHeight
    }
    const ctx = c.getContext('2d')
    clearOverlay(ctx)
    if (latestCaptureLandmarks) {
      drawHandSkeleton(ctx, latestCaptureLandmarks, {
        width: c.width,
        height: c.height,
      })
    }
  })

  const hands = getHandsInstance()
  const loop = async () => {
    if (captureVideoRef.value && captureVideoRef.value.readyState >= 2) {
      await hands.send({ image: captureVideoRef.value })
    }
    if (cameraOpen.value) {
      captureLoopId = requestAnimationFrame(loop)
    }
  }
  captureLoopId = requestAnimationFrame(loop)
}

function captureSingle() {
  if (!latestCaptureLandmarks || cameraIndex.value < 0) return
  const cls = classes.value[cameraIndex.value]
  if (!cls) return
  const norm = normalizeLandmarks(latestCaptureLandmarks)
  if (!norm) return
  cls.capturedLandmarks.push(norm)
  cls.landmarkCount++
  cls.attemptCount++
  captureCollected.value++

  // Snapshot thumbnail with skeleton
  const v = captureVideoRef.value
  if (v) {
    const cap = document.createElement('canvas')
    cap.width = v.videoWidth
    cap.height = v.videoHeight
    const cctx = cap.getContext('2d')
    cctx.drawImage(v, 0, 0)
    drawHandSkeleton(cctx, latestCaptureLandmarks, {
      width: cap.width,
      height: cap.height,
    })
    pushThumb(cls, cap, true)
  }
}

function captureBurst(target) {
  if (captureBurstActive.value) return
  captureBurstActive.value = true
  let count = 0
  burstTimerId = setInterval(() => {
    if (!cameraOpen.value || count >= target) {
      stopBurst()
      return
    }
    if (latestCaptureLandmarks) {
      captureSingle()
      count++
    }
  }, 200)
}

function stopBurst() {
  captureBurstActive.value = false
  if (burstTimerId) {
    clearInterval(burstTimerId)
    burstTimerId = null
  }
}

function closeCamera() {
  stopBurst()
  if (captureLoopId) {
    cancelAnimationFrame(captureLoopId)
    captureLoopId = null
  }
  if (captureStream) {
    captureStream.getTracks().forEach((t) => t.stop())
    captureStream = null
  }
  if (captureVideoRef.value) captureVideoRef.value.srcObject = null
  latestCaptureLandmarks = null
  cameraOpen.value = false
}

// —— Augmentation ——
function augmentSample(landmark) {
  const out = [landmark]
  // jitter
  const jitter = landmark.map((v, i) => {
    if (i < 3) return v // wrist stays at origin
    return v + (Math.random() - 0.5) * 0.02
  })
  out.push(jitter)
  // mirror flip x (every 3rd index)
  const mirror = landmark.map((v, i) => (i % 3 === 0 ? -v : v))
  out.push(mirror)
  return out
}

// —— Train ——
async function trainModel() {
  if (isTraining.value) return
  isTraining.value = true
  progressText.value = ''
  lastResult.value = null

  const log = (line) => {
    progressText.value += (progressText.value ? '\n' : '') + line
  }

  try {
    log('📦 รวบรวมตัวอย่างที่สกัดไว้แล้วระหว่างอัปโหลด/ถ่ายกล้อง...')
    const newSamples = []

    for (const cls of classes.value) {
      if (!cls.name) continue
      // Files were already processed on upload (uploadedSamples), so we
      // do NOT re-run MediaPipe here. Anything captured from camera is
      // already in capturedLandmarks.
      for (const norm of cls.uploadedSamples) {
        newSamples.push({ label: cls.name, landmark: norm })
      }
      for (const norm of cls.capturedLandmarks) {
        newSamples.push({ label: cls.name, landmark: norm })
      }
    }

    log(`✅ ตัวอย่างพร้อมเทรน ${newSamples.length} samples`)

    // Combine with replay buffer (only in finetune mode + same normalize protocol)
    let combined = newSamples
    if (trainingMode.value === 'finetune' && baseDataset.value.length) {
      if (baseModelInfo.value?.normalized) {
        combined = [...baseDataset.value, ...newSamples]
        log(`📚 รวมข้อมูลสะสมเดิม ${baseDataset.value.length} samples → รวม ${combined.length} samples`)
      } else {
        log(`⚠️ โมเดลเดิมใช้ feature เก่า — ข้อมูลสะสมเดิมเข้ากันไม่ได้ จะใช้แต่ข้อมูลใหม่`)
      }
    }

    if (!combined.length) {
      log('❌ ไม่มีข้อมูลฝึก')
      isTraining.value = false
      return
    }

    // Optional augmentation
    let trainSet = combined
    if (useAugmentation.value) {
      trainSet = []
      for (const s of combined) {
        for (const aug of augmentSample(s.landmark)) {
          trainSet.push({ label: s.label, landmark: aug })
        }
      }
      log(`🔧 augmentation: ${combined.length} → ${trainSet.length} samples`)
    }

    // Build label set from combined (so finetune keeps old classes)
    const labelSet = Array.from(new Set(combined.map((d) => d.label)))
    labelSet.sort()
    log(`🏷️ คลาสทั้งหมด ${labelSet.length}: ${labelSet.join(', ')}`)

    // Shuffle
    for (let i = trainSet.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[trainSet[i], trainSet[j]] = [trainSet[j], trainSet[i]]
    }

    const xs = tf.tensor2d(trainSet.map((d) => d.landmark))
    const ysIdx = trainSet.map((d) => labelSet.indexOf(d.label))
    const ys = tf.oneHot(tf.tensor1d(ysIdx, 'int32'), labelSet.length)

    // Build new model (always, to ensure consistent normalize protocol)
    const inputDim = trainSet[0].landmark.length
    const model = tf.sequential()
    model.add(tf.layers.dense({ inputShape: [inputDim], units: 256, activation: 'relu' }))
    model.add(tf.layers.batchNormalization())
    model.add(tf.layers.dropout({ rate: 0.3 }))
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }))
    model.add(tf.layers.dropout({ rate: 0.3 }))
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }))
    model.add(tf.layers.dense({ units: labelSet.length, activation: 'softmax' }))
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    })

    log(`🧠 เริ่มเทรน (input dim ${inputDim}, ${labelSet.length} classes)...`)

    const samplesPerClass = labelSet.map(
      (l) => combined.filter((s) => s.label === l).length
    )
    const minClassCount = Math.min(...samplesPerClass)
    const useValSplit = minClassCount >= 8 && trainSet.length >= 50
    if (!useValSplit) {
      log(`⚠️ ตัวอย่างน้อยเกินจะแยก validation set (น้อยสุด ${minClassCount}/class) — train ทั้งชุด`)
    }

    let bestValLoss = Infinity
    let lastEpochLogs = {}
    const fitConfig = {
      epochs: 80,
      batchSize: 32,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          lastEpochLogs = logs
          if (logs.val_loss != null && logs.val_loss < bestValLoss) {
            bestValLoss = logs.val_loss
          }
          const acc = (logs.acc ?? logs.accuracy) * 100
          const vAcc = (logs.val_acc ?? logs.val_accuracy)
          const vAccStr = vAcc != null ? ` val_acc=${(vAcc * 100).toFixed(2)}%` : ''
          log(`Epoch ${epoch + 1}: acc=${acc.toFixed(2)}% loss=${logs.loss.toFixed(4)}${vAccStr}`)
        },
      },
    }
    if (useValSplit) {
      fitConfig.validationSplit = 0.2
      fitConfig.callbacks = {
        ...fitConfig.callbacks,
        ...{ earlyStopping: null },
      }
    }

    // Manual early stopping using val_loss
    let patience = 0
    let stop = false
    const earlyStopThreshold = 10
    fitConfig.callbacks.onEpochEnd = ((origOnEpochEnd) => (epoch, logs) => {
      origOnEpochEnd(epoch, logs)
      if (useValSplit && logs.val_loss != null) {
        if (logs.val_loss > bestValLoss) {
          patience++
          if (patience >= earlyStopThreshold) {
            stop = true
            log(`⏹ Early stop ที่ epoch ${epoch + 1} (val_loss ไม่ดีขึ้น ${earlyStopThreshold} epoch)`)
            model.stopTraining = true
          }
        } else {
          patience = 0
        }
      }
    })(fitConfig.callbacks.onEpochEnd)

    await model.fit(xs, ys, fitConfig)
    xs.dispose()
    ys.dispose()

    const finalAcc = lastEpochLogs.acc ?? lastEpochLogs.accuracy
    const finalLoss = lastEpochLogs.loss
    const finalValAcc = lastEpochLogs.val_acc ?? lastEpochLogs.val_accuracy ?? finalAcc
    const finalValLoss = lastEpochLogs.val_loss ?? finalLoss

    log(`💾 กำลังบันทึกโมเดล + dataset ลงเซิร์ฟเวอร์...`)
    let savedData = null
    await model.save(
      tf.io.withSaveHandler(async (artifacts) => {
        savedData = {
          modelTopology: artifacts.modelTopology,
          weightSpecs: artifacts.weightSpecs,
          weightData: Array.from(new Uint8Array(artifacts.weightData)),
        }
      })
    )

    const saveRes = await fetch('/api/save-model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: savedData,
        labels: labelSet,
        dataset: combined, // store combined, NOT augmented
        metrics: {
          accuracy: finalAcc,
          loss: finalLoss,
          valAccuracy: finalValAcc,
          valLoss: finalValLoss,
          normalized: true,
          datasetSize: combined.length,
          epochs: 80,
          baseModel:
            trainingMode.value === 'finetune' ? selectedVersion.value : null,
        },
      }),
    })

    if (!saveRes.ok) throw new Error(await saveRes.text())
    const saveJson = await saveRes.json()

    const baseline = baseModelInfo.value?.accuracy ?? null
    const delta =
      baseline != null && finalValAcc != null ? finalValAcc - baseline : null
    lastResult.value = {
      path: saveJson.path,
      valAccuracy: finalValAcc,
      baselineAccuracy: baseline,
      delta,
      datasetSize: combined.length,
    }
    log(`✅ บันทึกเสร็จ: ${saveJson.path}`)

    await loadAvailableVersions()
  } catch (err) {
    console.error('train error:', err)
    log(`❌ เกิดข้อผิดพลาด: ${err.message || err}`)
  } finally {
    isTraining.value = false
  }
}

onMounted(async () => {
  await loadAvailableVersions()
})

onBeforeUnmount(() => {
  closeCamera()
})
</script>

<style scoped>
.train-page {
  padding: 16px;
  max-width: 1100px;
  margin: 0 auto;
  font-family: system-ui, sans-serif;
  text-align: left;
}
h1 {
  font-size: 1.4rem;
  margin: 0 0 12px;
}
.card {
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 6px;
  padding: 10px 12px;
  margin-bottom: 12px;
}
.block-label {
  display: block;
  font-weight: 600;
  margin-bottom: 6px;
}
.radio {
  display: inline-block;
  margin-right: 16px;
  cursor: pointer;
}
.version-select {
  padding: 6px 8px;
  border-radius: 4px;
  border: 1px solid #888;
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
.warn {
  color: #d97706;
  margin-top: 4px;
}
.warn-text {
  color: #d97706;
  margin-left: 6px;
  font-size: 0.85em;
}
.proc-row {
  font-size: 0.85em;
  color: #3b82f6;
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 4px;
}
.proc-row .spin {
  animation: spin 1.5s linear infinite;
  display: inline-block;
}
@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
.picker-grid {
  display: grid;
  grid-template-columns: repeat(11, 1fr);
  gap: 4px;
  margin: 8px 0;
}
@media (max-width: 640px) {
  .picker-grid {
    grid-template-columns: repeat(7, 1fr);
  }
}
.picker-btn {
  position: relative;
  padding: 6px 0;
  border: 1px solid #888;
  background: #f0f0f0;
  color: #333;
  border-radius: 4px;
  cursor: pointer;
  font-size: 1.1rem;
  min-height: 36px;
}
.picker-btn.active {
  background: #3b82f6;
  color: #fff;
  border-color: #2563eb;
}
.picker-btn.has-data {
  background: #10b981;
  color: #fff;
  border-color: #059669;
}
.picker-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
.picker-btn .ch {
  display: inline-block;
}
.picker-btn .badge {
  position: absolute;
  top: -4px;
  right: -4px;
  font-size: 0.65em;
  background: #fff;
  color: #111;
  border-radius: 8px;
  padding: 1px 4px;
  border: 1px solid #ddd;
}
.btn-secondary {
  padding: 6px 12px;
  border: 1px solid #888;
  background: #fff;
  color: #111;
  border-radius: 4px;
  cursor: pointer;
  margin-top: 4px;
}
.class-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 10px;
  margin-bottom: 12px;
}
.class-block {
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 6px;
  padding: 10px;
}
.class-head {
  display: flex;
  gap: 6px;
  align-items: center;
  margin-bottom: 6px;
}
.class-name {
  flex: 1;
  padding: 4px 6px;
  border: 1px solid #888;
  border-radius: 4px;
  font-size: 1rem;
  background: #fff;
  color: #111;
}
.x-btn {
  background: transparent;
  border: 1px solid #888;
  color: inherit;
  border-radius: 4px;
  cursor: pointer;
  font-size: 1rem;
  width: 28px;
  height: 28px;
  line-height: 1;
}
.counter {
  font-size: 0.9em;
  margin-bottom: 6px;
}
.input-row {
  margin-bottom: 6px;
  font-size: 0.9em;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
}
.file-label {
  cursor: pointer;
}
.btn-camera {
  background: #6366f1;
  color: #fff;
  border: none;
  padding: 4px 10px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.9em;
}
.btn-camera:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
.filename {
  font-size: 0.8em;
  color: #aaa;
}
.thumb-strip {
  display: flex;
  gap: 4px;
  overflow-x: auto;
  margin-top: 6px;
}
.thumb {
  position: relative;
  border: 2px solid transparent;
  border-radius: 4px;
  flex: 0 0 auto;
  cursor: pointer;
  transition: transform 0.1s ease;
}
.thumb:hover, .thumb:active {
  transform: scale(1.06);
}
.thumb img {
  display: block;
  max-height: 60px;
  height: 60px;
  width: auto;
}
.thumb.detected {
  border-color: #10b981;
}
.thumb.not-detected {
  border-color: #ef4444;
  opacity: 0.7;
}
.thumb-tag {
  position: absolute;
  bottom: 2px;
  left: 2px;
  font-size: 0.65em;
  background: rgba(239, 68, 68, 0.85);
  color: #fff;
  padding: 0 4px;
  border-radius: 2px;
}
.checkbox {
  display: block;
  margin-bottom: 8px;
}
.btn-train {
  padding: 10px 18px;
  background: #2563eb;
  color: #fff;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 1rem;
}
.btn-train:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.progress {
  background: #111;
  color: #0f0;
  padding: 8px;
  border-radius: 4px;
  font-family: monospace;
  font-size: 0.85em;
  max-height: 300px;
  overflow: auto;
  white-space: pre-wrap;
}
.result-box {
  border-left: 3px solid #10b981;
}
.good {
  color: #10b981;
  font-weight: 600;
}
.bad {
  color: #ef4444;
  font-weight: 600;
}

/* Modal */
.modal-bg {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}
.modal {
  background: #1a1a1a;
  color: #eee;
  padding: 14px;
  border-radius: 8px;
  max-width: 95vw;
  min-width: 320px;
  border: 1px solid #444;
}
.modal-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.thumb-viewer {
  background: #1a1a1a;
  color: #eee;
  padding: 12px;
  border-radius: 8px;
  max-width: 95vw;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  gap: 8px;
  border: 1px solid #444;
}
.thumb-viewer-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.thumb-viewer-img {
  max-width: 100%;
  max-height: 75vh;
  width: auto;
  height: auto;
  object-fit: contain;
  border-radius: 4px;
  background: #000;
}

/* Webcam overlay shared */
.mirror {
  transform: scaleX(-1);
}
.webcam-stack {
  position: relative;
  display: inline-block;
  line-height: 0;
  border: 1px solid #555;
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
.capture-controls {
  margin-top: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
}
.capture-status {
  width: 100%;
}
.btn-capture {
  background: #16a34a;
  color: #fff;
  border: none;
  padding: 6px 12px;
  border-radius: 4px;
  cursor: pointer;
}
.btn-capture-stop {
  background: #ef4444;
  color: #fff;
  border: none;
  padding: 6px 12px;
  border-radius: 4px;
  cursor: pointer;
}
</style>
