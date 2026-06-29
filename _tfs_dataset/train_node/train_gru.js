/**
 * Step 1 of the /exp pipeline.
 *
 * Trains a small GRU sequence classifier on PSEUDO-SEQUENCES generated from
 * the 2048 static landmark samples we already collected. Each static sample
 * is replicated SEQ_LEN=15 times with light Gaussian noise to fake the
 * temporal structure a sequence model expects, so we can prove the
 * architecture + deployment pipeline before collecting real video clips.
 *
 * Output is the standard TFJS Layers format (model.json + weights.bin)
 * plus labels.json, metrics.json (incl. sequenceLength + architecture),
 * and the original NON-augmented landmarks as dataset.json so the
 * replay-buffer mechanism keeps working.
 */
import * as tf from '@tensorflow/tfjs'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const DATASET_PATH = path.resolve(__dirname, '..', 'extracted.json')
const OUT_DIR = path.resolve(__dirname, '..', 'trained_gru_model')

const SEQ_LEN = 15           // pseudo-sequence length in frames
const NOISE_SIGMA = 0.005    // per-coordinate Gaussian noise (normalized space)

function gauss() {
  // Box-Muller, used twice gives ~N(0,1)
  let u = 0, v = 0
  while (u === 0) u = Math.random()
  while (v === 0) v = Math.random()
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)
}

/** Replicate a single landmark vector into a noisy sequence.
 *  Wrist (idx 0..2 after the wrist-relative normalize) is always [0,0,0],
 *  so don't perturb those — they encode the centering invariant.
 */
function makePseudoSequence(flatLandmark) {
  const seq = new Array(SEQ_LEN)
  for (let t = 0; t < SEQ_LEN; t++) {
    const frame = new Array(flatLandmark.length)
    for (let i = 0; i < flatLandmark.length; i++) {
      if (i < 3) {
        frame[i] = flatLandmark[i] // wrist stays at origin
      } else {
        frame[i] = flatLandmark[i] + gauss() * NOISE_SIGMA
      }
    }
    seq[t] = frame
  }
  return seq
}

async function main() {
  await tf.ready()
  console.log('TF.js backend:', tf.getBackend())

  const raw = JSON.parse(fs.readFileSync(DATASET_PATH, 'utf-8'))
  const dataset = raw.dataset
  console.log(`Loaded ${dataset.length} static landmark samples`)

  const labelSet = Array.from(new Set(dataset.map((s) => s.label))).sort()
  console.log('Classes:', labelSet)
  const perClass = Object.fromEntries(
    labelSet.map((l) => [l, dataset.filter((s) => s.label === l).length])
  )
  console.log('Per-class counts:', perClass)

  const inputDim = dataset[0].landmark.length

  // Build pseudo-sequence training set with augmentation per the existing
  // production protocol (jitter + horizontal flip), expanded into sequences.
  function augmentStatic(flat) {
    const out = [flat]
    const jitter = flat.map((v, i) =>
      i < 3 ? v : v + (Math.random() - 0.5) * 0.02
    )
    out.push(jitter)
    const flip = flat.map((v, i) => (i % 3 === 0 ? -v : v))
    out.push(flip)
    return out
  }

  const sequences = []
  const seqLabels = []
  for (const s of dataset) {
    for (const aug of augmentStatic(s.landmark)) {
      sequences.push(makePseudoSequence(aug))
      seqLabels.push(s.label)
    }
  }
  console.log(`Pseudo-sequence training pool: ${sequences.length} clips of length ${SEQ_LEN}`)

  // Shuffle in-place (Fisher-Yates) keeping labels aligned
  for (let i = sequences.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[sequences[i], sequences[j]] = [sequences[j], sequences[i]]
    ;[seqLabels[i], seqLabels[j]] = [seqLabels[j], seqLabels[i]]
  }

  const xs = tf.tensor3d(sequences, [sequences.length, SEQ_LEN, inputDim])
  const ysIdx = seqLabels.map((l) => labelSet.indexOf(l))
  const ys = tf.oneHot(tf.tensor1d(ysIdx, 'int32'), labelSet.length)

  // Two-layer GRU per Dr.Anan / Gemini recommendation.
  // params ~ 3*(63*64 + 64*64 + 64) + 3*(64*64 + 64*64 + 64) + 64*8
  //       ~ 24k + 25k + 0.5k = ~50k -> well under 5MB target.
  const model = tf.sequential()
  model.add(tf.layers.gru({
    inputShape: [SEQ_LEN, inputDim],
    units: 64,
    returnSequences: true,
  }))
  model.add(tf.layers.gru({
    units: 64,
    returnSequences: false,
  }))
  model.add(tf.layers.dropout({ rate: 0.3 }))
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }))
  model.add(tf.layers.dense({ units: labelSet.length, activation: 'softmax' }))
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  })
  console.log(`Model param count: ${model.countParams()}`)

  let bestValLoss = Infinity
  let patience = 0
  const earlyStopPatience = 10
  let lastLogs = {}

  await model.fit(xs, ys, {
    epochs: 60,
    batchSize: 32,
    shuffle: true,
    validationSplit: 0.2,
    verbose: 0,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        lastLogs = logs
        const acc = ((logs.acc ?? logs.accuracy) * 100).toFixed(2)
        const vAcc = ((logs.val_acc ?? logs.val_accuracy) * 100).toFixed(2)
        console.log(
          `Epoch ${String(epoch + 1).padStart(3)}: ` +
          `acc=${acc}% loss=${logs.loss.toFixed(4)} ` +
          `val_acc=${vAcc}% val_loss=${logs.val_loss.toFixed(4)}`
        )
        if (logs.val_loss < bestValLoss) {
          bestValLoss = logs.val_loss
          patience = 0
        } else {
          patience++
          if (patience >= earlyStopPatience) {
            console.log(`Early stop at epoch ${epoch + 1} (val_loss did not improve for ${earlyStopPatience} epochs)`)
            model.stopTraining = true
          }
        }
      },
    },
  })

  xs.dispose()
  ys.dispose()

  if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true })

  let artifactJson, artifactWeights
  await model.save(
    tf.io.withSaveHandler(async (artifacts) => {
      artifactJson = {
        modelTopology: artifacts.modelTopology,
        weightsManifest: [
          { paths: ['group1-shard1of1.bin'], weights: artifacts.weightSpecs },
        ],
      }
      artifactWeights = Buffer.from(artifacts.weightData)
      return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } }
    })
  )

  fs.writeFileSync(path.join(OUT_DIR, 'model.json'), JSON.stringify(artifactJson))
  fs.writeFileSync(path.join(OUT_DIR, 'group1-shard1of1.bin'), artifactWeights)
  // Namespaced labels per the agreed forward-compat schema
  const namespacedLabels = labelSet.map((l) => `consonant_${l}`)
  fs.writeFileSync(path.join(OUT_DIR, 'labels.json'), JSON.stringify(namespacedLabels, null, 2))
  fs.writeFileSync(
    path.join(OUT_DIR, 'dataset.json'),
    JSON.stringify(dataset.map((s) => ({ label: s.label, landmark: s.landmark })))
  )

  const metrics = {
    accuracy: lastLogs.acc ?? lastLogs.accuracy,
    loss: lastLogs.loss,
    valAccuracy: lastLogs.val_acc ?? lastLogs.val_accuracy,
    valLoss: lastLogs.val_loss,
    normalized: true,
    datasetSize: dataset.length,
    epochs: 60,
    baseModel: null,
    architecture: 'GRU64x2-pseudoseq',
    sequenceLength: SEQ_LEN,
    inputDim,
    bootstrappedFrom: 'pseudo-sequences (replicate static frame + Gaussian noise σ=0.005)',
    sourcedFrom: 'Mendeley One-Stage-TFS (บ) + Mendeley ASL-HG (others via mapped letter)',
  }
  fs.writeFileSync(path.join(OUT_DIR, 'metrics.json'), JSON.stringify(metrics, null, 2))

  console.log('\nSaved exp GRU artifacts to:', OUT_DIR)
  console.log('Final metrics:', JSON.stringify(metrics, null, 2))
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
