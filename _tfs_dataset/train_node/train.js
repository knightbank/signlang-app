import * as tf from '@tensorflow/tfjs'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const DATASET_PATH = path.resolve(__dirname, '..', 'extracted.json')
const OUT_DIR = path.resolve(__dirname, '..', 'trained_model')

async function main() {
  console.log('TF.js backend:', tf.getBackend())
  await tf.ready()

  const raw = JSON.parse(fs.readFileSync(DATASET_PATH, 'utf-8'))
  const dataset = raw.dataset
  console.log(`Loaded ${dataset.length} samples`)

  // Build label set (sorted for determinism)
  const labelSet = Array.from(new Set(dataset.map((s) => s.label))).sort()
  console.log('Classes:', labelSet)
  const classCount = Object.fromEntries(
    labelSet.map((l) => [l, dataset.filter((s) => s.label === l).length])
  )
  console.log('Per-class counts:', classCount)

  // Build augmented training set: original + jitter + horizontal-flip
  function augment(landmark) {
    const out = [landmark]
    // jitter (skip wrist idx 0 which is at origin after normalize)
    const jitter = landmark.map((v, i) => {
      if (i < 3) return v
      return v + (Math.random() - 0.5) * 0.02
    })
    out.push(jitter)
    // horizontal flip (x is every 3rd index)
    const flip = landmark.map((v, i) => (i % 3 === 0 ? -v : v))
    out.push(flip)
    return out
  }

  const trainSet = []
  for (const s of dataset) {
    for (const aug of augment(s.landmark)) {
      trainSet.push({ label: s.label, landmark: aug })
    }
  }
  console.log(`Augmented training pool: ${trainSet.length} samples`)

  // Shuffle
  for (let i = trainSet.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[trainSet[i], trainSet[j]] = [trainSet[j], trainSet[i]]
  }

  const inputDim = trainSet[0].landmark.length
  const xs = tf.tensor2d(trainSet.map((d) => d.landmark))
  const ysIdx = trainSet.map((d) => labelSet.indexOf(d.label))
  const ys = tf.oneHot(tf.tensor1d(ysIdx, 'int32'), labelSet.length)

  // Model architecture matches TrainingPage.vue
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

  console.log(`Training ${classCount ? '' : ''}— input=${inputDim} classes=${labelSet.length} params=${model.countParams()}`)

  let bestValLoss = Infinity
  let patience = 0
  const earlyStopPatience = 12
  let lastLogs = {}

  await model.fit(xs, ys, {
    epochs: 100,
    batchSize: 32,
    shuffle: true,
    validationSplit: 0.2,
    verbose: 0,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        lastLogs = logs
        const acc = ((logs.acc ?? logs.accuracy) * 100).toFixed(2)
        const vAcc = ((logs.val_acc ?? logs.val_accuracy) * 100).toFixed(2)
        const loss = logs.loss.toFixed(4)
        const vLoss = logs.val_loss.toFixed(4)
        console.log(`Epoch ${String(epoch + 1).padStart(3)}: acc=${acc}% loss=${loss}  val_acc=${vAcc}% val_loss=${vLoss}`)

        if (logs.val_loss < bestValLoss) {
          bestValLoss = logs.val_loss
          patience = 0
        } else {
          patience++
          if (patience >= earlyStopPatience) {
            console.log(`Early stopping at epoch ${epoch + 1} (val_loss did not improve for ${earlyStopPatience} epochs)`)
            model.stopTraining = true
          }
        }
      },
    },
  })

  xs.dispose()
  ys.dispose()

  // Save model in TFJS layers format
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
  fs.writeFileSync(path.join(OUT_DIR, 'labels.json'), JSON.stringify(labelSet, null, 2))
  // Save the ORIGINAL (non-augmented) dataset for replay buffer
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
    epochs: 100,
    baseModel: null,
    sourcedFrom: 'Mendeley One-Stage-TFS (บ) + Mendeley ASL-HG (others via mapped letter)',
  }
  fs.writeFileSync(path.join(OUT_DIR, 'metrics.json'), JSON.stringify(metrics, null, 2))

  console.log('\nSaved artifacts to:', OUT_DIR)
  console.log('Final metrics:', JSON.stringify(metrics, null, 2))
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
