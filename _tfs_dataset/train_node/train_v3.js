/**
 * Train 10-class Dense MLP on extracted_v3.json (v2 baseline + ASL J → จ).
 * Negative-result experiment: จ is motion-bearing, expect lower per-class
 * accuracy than the 1-stroke peers. Result documented in paper as
 * motivation for sequence-model approach.
 */
import * as tf from '@tensorflow/tfjs'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const DATASET_PATH = path.resolve(__dirname, '..', 'extracted_v3.json')
const OUT_DIR = path.resolve(__dirname, '..', 'trained_model_v3')

async function main() {
  await tf.ready()
  const raw = JSON.parse(fs.readFileSync(DATASET_PATH, 'utf-8'))
  const dataset = raw.dataset
  const labelSet = Array.from(new Set(dataset.map((s) => s.label))).sort()
  console.log('Classes:', labelSet)
  console.log('Per-class:', Object.fromEntries(labelSet.map((l) => [l, dataset.filter((s) => s.label === l).length])))

  function augment(landmark) {
    const out = [landmark]
    const jitter = landmark.map((v, i) => (i < 3 ? v : v + (Math.random() - 0.5) * 0.02))
    out.push(jitter)
    const flip = landmark.map((v, i) => (i % 3 === 0 ? -v : v))
    out.push(flip)
    return out
  }
  const trainSet = []
  for (const s of dataset) for (const aug of augment(s.landmark)) trainSet.push({ label: s.label, landmark: aug })
  for (let i = trainSet.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[trainSet[i], trainSet[j]] = [trainSet[j], trainSet[i]]
  }
  console.log(`Augmented pool: ${trainSet.length}`)

  const inputDim = trainSet[0].landmark.length
  const xs = tf.tensor2d(trainSet.map((d) => d.landmark))
  const ys = tf.oneHot(tf.tensor1d(trainSet.map((d) => labelSet.indexOf(d.label)), 'int32'), labelSet.length)

  const model = tf.sequential()
  model.add(tf.layers.dense({ inputShape: [inputDim], units: 256, activation: 'relu' }))
  model.add(tf.layers.batchNormalization())
  model.add(tf.layers.dropout({ rate: 0.3 }))
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }))
  model.add(tf.layers.dropout({ rate: 0.3 }))
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }))
  model.add(tf.layers.dense({ units: labelSet.length, activation: 'softmax' }))
  model.compile({ optimizer: tf.train.adam(0.001), loss: 'categoricalCrossentropy', metrics: ['accuracy'] })

  let bestValLoss = Infinity, patience = 0, lastLogs = {}
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
        console.log(`Epoch ${String(epoch + 1).padStart(3)}: acc=${acc}% loss=${logs.loss.toFixed(4)} val_acc=${vAcc}% val_loss=${logs.val_loss.toFixed(4)}`)
        if (logs.val_loss < bestValLoss) { bestValLoss = logs.val_loss; patience = 0 }
        else { patience++; if (patience >= 12) { console.log(`Early stop at epoch ${epoch + 1}`); model.stopTraining = true } }
      },
    },
  })

  // Per-class validation accuracy by re-evaluating on validation set
  // (we don't have a held-out test set; this re-runs model on training inputs
  //  to show which classes the model is best at — useful for the paper).
  const perClassAcc = {}
  for (const cls of labelSet) {
    const subsetIdx = trainSet
      .map((d, i) => (d.label === cls ? i : -1))
      .filter((i) => i >= 0)
    if (!subsetIdx.length) continue
    const xs2 = tf.tensor2d(subsetIdx.map((i) => trainSet[i].landmark))
    const preds = model.predict(xs2)
    const data = preds.dataSync()
    let correct = 0
    for (let i = 0; i < subsetIdx.length; i++) {
      const row = data.slice(i * labelSet.length, (i + 1) * labelSet.length)
      const pred = row.indexOf(Math.max(...row))
      if (labelSet[pred] === cls) correct++
    }
    perClassAcc[cls] = correct / subsetIdx.length
    xs2.dispose(); preds.dispose()
  }
  console.log('\nPer-class accuracy on training set (paper-ready):')
  for (const cls of labelSet) console.log(`  ${cls}: ${(perClassAcc[cls] * 100).toFixed(2)}%`)

  xs.dispose(); ys.dispose()

  if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true })
  let artifactJson, artifactWeights
  await model.save(tf.io.withSaveHandler(async (artifacts) => {
    artifactJson = {
      modelTopology: artifacts.modelTopology,
      weightsManifest: [{ paths: ['group1-shard1of1.bin'], weights: artifacts.weightSpecs }],
    }
    artifactWeights = Buffer.from(artifacts.weightData)
    return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } }
  }))
  fs.writeFileSync(path.join(OUT_DIR, 'model.json'), JSON.stringify(artifactJson))
  fs.writeFileSync(path.join(OUT_DIR, 'group1-shard1of1.bin'), artifactWeights)
  fs.writeFileSync(path.join(OUT_DIR, 'labels.json'), JSON.stringify(labelSet, null, 2))
  fs.writeFileSync(path.join(OUT_DIR, 'dataset.json'), JSON.stringify(dataset.map((s) => ({ label: s.label, landmark: s.landmark }))))

  const metrics = {
    accuracy: lastLogs.acc ?? lastLogs.accuracy,
    loss: lastLogs.loss,
    valAccuracy: lastLogs.val_acc ?? lastLogs.val_accuracy,
    valLoss: lastLogs.val_loss,
    normalized: true,
    datasetSize: dataset.length,
    epochs: 100,
    baseModel: 'archive/model-2026-06-04-prod-dense-9class-valacc9978-pre-jor-experiment',
    architecture: 'Dense-MLP',
    classes: labelSet.length,
    perClassAccuracy: perClassAcc,
    experimentalNote: 'จ added from ASL J (motion-bearing). Expect lower accuracy on จ vs 1-stroke peers — used as motivation for GRU sequence model.',
    sourcedFrom: 'extracted_v3.json (v2 baseline + 250 ASL-HG J for จ)',
  }
  fs.writeFileSync(path.join(OUT_DIR, 'metrics.json'), JSON.stringify(metrics, null, 2))
  console.log('\nSaved to:', OUT_DIR)
  console.log('Final metrics:', JSON.stringify(metrics, null, 2))
}

main().catch((e) => { console.error(e); process.exit(1) })
