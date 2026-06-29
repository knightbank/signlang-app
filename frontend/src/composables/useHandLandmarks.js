import { Hands, HAND_CONNECTIONS } from '@mediapipe/hands'

export { HAND_CONNECTIONS }

const VIDEO_OPTIONS = {
  maxNumHands: 1,
  modelComplexity: 1,
  staticImageMode: false,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.5,
}

const STATIC_OPTIONS = {
  maxNumHands: 1,
  modelComplexity: 1,
  staticImageMode: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
}

const STATIC_MAX_DIM = 640

let _videoHands = null
let _staticHands = null
let _videoHandler = null
let _staticHandler = null

function locateFile(file) {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
}

export function getHandsInstance() {
  if (!_videoHands) {
    _videoHands = new Hands({ locateFile })
    _videoHands.setOptions(VIDEO_OPTIONS)
    _videoHands.onResults((results) => {
      if (_videoHandler) _videoHandler(results)
    })
  }
  return _videoHands
}

function getStaticHandsInstance() {
  if (!_staticHands) {
    _staticHands = new Hands({ locateFile })
    _staticHands.setOptions(STATIC_OPTIONS)
    _staticHands.onResults((results) => {
      if (_staticHandler) _staticHandler(results)
    })
  }
  return _staticHands
}

export function setOnResults(handler) {
  _videoHandler = handler
  getHandsInstance()
}

// Camera/video streams: use the tracking-tuned instance and the shared
// _videoHandler so an outer rAF loop can mutate it freely.
export function handsSend(canvasOrImageOrVideo) {
  const hands = getHandsInstance()
  return new Promise((resolve) => {
    _videoHandler = (results) => resolve(results)
    hands.send({ image: canvasOrImageOrVideo }).catch(() => {
      resolve({ multiHandLandmarks: [] })
    })
  })
}

// Static images / extracted video frames: use the static-mode instance with
// a lower detection threshold. Auto-downsamples huge phone photos so
// MediaPipe does not run out of memory on a single tab.
export function handsSendStatic(sourceCanvasOrImage) {
  const hands = getStaticHandsInstance()
  const downsampled = ensureReasonableSize(sourceCanvasOrImage)
  return new Promise((resolve) => {
    _staticHandler = (results) => resolve(results)
    hands.send({ image: downsampled }).catch(() => {
      resolve({ multiHandLandmarks: [] })
    })
  })
}

function ensureReasonableSize(src) {
  const w = src.width || src.videoWidth || src.naturalWidth || 0
  const h = src.height || src.videoHeight || src.naturalHeight || 0
  if (!w || !h) return src
  if (Math.max(w, h) <= STATIC_MAX_DIM) return src
  const scale = STATIC_MAX_DIM / Math.max(w, h)
  const cw = Math.round(w * scale)
  const ch = Math.round(h * scale)
  const canvas = document.createElement('canvas')
  canvas.width = cw
  canvas.height = ch
  canvas.getContext('2d').drawImage(src, 0, 0, cw, ch)
  return canvas
}

export function normalizeLandmarks(raw) {
  if (!raw || raw.length !== 21) return null
  const wrist = raw[0]
  const middleMCP = raw[9]
  const dx = middleMCP.x - wrist.x
  const dy = middleMCP.y - wrist.y
  const dz = middleMCP.z - wrist.z
  const scale = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1
  const flat = new Array(63)
  for (let i = 0; i < 21; i++) {
    flat[i * 3] = (raw[i].x - wrist.x) / scale
    flat[i * 3 + 1] = (raw[i].y - wrist.y) / scale
    flat[i * 3 + 2] = (raw[i].z - wrist.z) / scale
  }
  return flat
}

export async function extractLandmarksFromCanvas(canvas) {
  const results = await handsSend(canvas)
  const raw = results.multiHandLandmarks?.[0] || null
  if (!raw) return { raw: null, flat: null, normalized: null }
  const flat = raw.flatMap((p) => [p.x, p.y, p.z])
  const normalized = normalizeLandmarks(raw)
  return { raw, flat, normalized }
}

export function drawHandSkeleton(ctx, landmarks, options = {}) {
  if (!landmarks || landmarks.length !== 21) return
  const {
    width = ctx.canvas.width,
    height = ctx.canvas.height,
    color = '#00ff88',
    dotColor = '#ffffff',
    wristColor = '#ff3366',
    lineWidth = 3,
    dotRadius = 4,
    wristRadius = 8,
  } = options

  ctx.lineWidth = lineWidth
  ctx.strokeStyle = color
  ctx.beginPath()
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.moveTo(landmarks[a].x * width, landmarks[a].y * height)
    ctx.lineTo(landmarks[b].x * width, landmarks[b].y * height)
  }
  ctx.stroke()

  ctx.fillStyle = dotColor
  for (let i = 1; i < 21; i++) {
    ctx.beginPath()
    ctx.arc(landmarks[i].x * width, landmarks[i].y * height, dotRadius, 0, Math.PI * 2)
    ctx.fill()
  }

  ctx.fillStyle = wristColor
  ctx.beginPath()
  ctx.arc(landmarks[0].x * width, landmarks[0].y * height, wristRadius, 0, Math.PI * 2)
  ctx.fill()
}

export function clearOverlay(ctx) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
}
