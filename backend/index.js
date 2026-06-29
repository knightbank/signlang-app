import express from 'express'
import path from 'path'
import { fileURLToPath } from 'url'
import { v4 as uuidv4 } from 'uuid'
import fs from 'fs-extra'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const app = express()
const PORT = 3001

app.get('/', (req, res) => {
    res.send('Server is running.');
});

app.use(express.json({ limit: '100mb' }))

const MODELS_DIR = path.join(__dirname, '../models')  
const DEV_DIR     = path.join(__dirname, 'models/dev')
const PROD_DIR    = path.join(__dirname, 'models/prod')
const ARCHIVE_DIR = path.join(__dirname, 'models/archive')

//app.use('/models', express.static(path.join(__dirname, '../models')))
app.use('/models', express.static(path.join(__dirname, 'models')))
//app.use('/models', express.static(MODELS_DIR))

// Ensure necessary directories exist
;[DEV_DIR, PROD_DIR, ARCHIVE_DIR].forEach(dir => fs.mkdirSync(dir, { recursive: true }))

function groupFor(version) {
  // Classify a model path into one of three buckets for the UI dropdown.
  //   active   = the currently-served model (prod/model-latest)
  //   rollback = previous-prod safety copies (.pre-rollback.*) and archive/
  //   dev      = work-in-progress saves from the training UI
  const parts = version.split('/')
  if (parts[0] === 'prod') {
    return parts[1] === 'model-latest' ? 'active' : 'rollback'
  }
  if (parts[0] === 'archive') return 'rollback'
  if (parts[0] === 'dev') return 'dev'
  return 'dev'
}

function readJsonSafe(p) {
  try { return JSON.parse(fs.readFileSync(p, 'utf-8')) }
  catch { return null }
}

// GET /api/models → return available model versions with metadata + group label
app.get('/api/models', (req, res) => {
  const listDir = (dir) => {
    const full = path.join(__dirname, 'models', dir)
    return fs.existsSync(full)
      ? fs.readdirSync(full).map(name => `${dir}/${name}`)
      : []
  }

  const versions = [
    ...listDir('prod'),
    ...listDir('dev'),
    ...listDir('archive')
  ]
    .filter(p => fs.existsSync(path.join(__dirname, 'models', p, 'labels.json')))
    .map((version) => {
      const dir = path.join(__dirname, 'models', version)
      const labels = readJsonSafe(path.join(dir, 'labels.json')) || []
      const metrics = readJsonSafe(path.join(dir, 'metrics.json')) || {}
      const dataset = readJsonSafe(path.join(dir, 'dataset.json'))
      const datasetSize = Array.isArray(dataset)
        ? dataset.length
        : (metrics.datasetSize ?? null)
      let createdAt = null
      try {
        createdAt = fs.statSync(path.join(dir, 'model.json')).mtime.toISOString()
      } catch {}
      return {
        version,
        group: groupFor(version),
        labels,
        labelsCount: labels.length,
        accuracy: metrics.accuracy ?? null,
        loss: metrics.loss ?? null,
        normalized: metrics.normalized === true,
        datasetSize,
        createdAt,
      }
    })
    .sort((a, b) => {
      // Group order: active > rollback > dev
      const groupOrder = { active: 0, rollback: 1, dev: 2 }
      const ag = groupOrder[a.group] ?? 99
      const bg = groupOrder[b.group] ?? 99
      if (ag !== bg) return ag - bg
      const aAcc = a.accuracy ?? -1
      const bAcc = b.accuracy ?? -1
      if (bAcc !== aAcc) return bAcc - aAcc
      return (b.createdAt || '').localeCompare(a.createdAt || '')
    })

  res.json(versions)
})

// GET /api/dataset/:scope/:name → return replay buffer for a model
app.get('/api/dataset/:scope/:name', (req, res) => {
  const { scope, name } = req.params
  const datasetPath = path.join(__dirname, 'models', scope, name, 'dataset.json')
  if (!fs.existsSync(datasetPath)) {
    return res.status(404).json({ error: 'dataset.json not found' })
  }
  res.sendFile(datasetPath)
})

// POST /api/save-model → save uploaded model + labels + metrics + replay buffer
app.post('/api/save-model', async (req, res) => {
  try {
    const { model, labels, metrics, dataset } = req.body;
    const now = new Date();
    const timestamp =
      now.getFullYear() + '-' +
      String(now.getMonth()+1).padStart(2,'0') + '-' +
      String(now.getDate()).padStart(2,'0') + 'T' +
      String(now.getHours()).padStart(2,'0') + '-' +
      String(now.getMinutes()).padStart(2,'0') + '-' +
      String(now.getSeconds()).padStart(2,'0') + '-local';
    const modelFolder = path.join(DEV_DIR, `model-${timestamp}`);
    fs.mkdirSync(modelFolder, { recursive: true });

    const jsonPath    = path.join(modelFolder, 'model.json');
    const binPath     = path.join(modelFolder, 'group1-shard1of1.bin');
    const labelsPath  = path.join(modelFolder, 'labels.json');
    const datasetPath = path.join(modelFolder, 'dataset.json');

    fs.writeFileSync(jsonPath, JSON.stringify({
      modelTopology: model.modelTopology,
      weightsManifest: [{ paths: ['group1-shard1of1.bin'], weights: model.weightSpecs }]
    }));
    fs.writeFileSync(binPath, Buffer.from(model.weightData));
    fs.writeFileSync(labelsPath, JSON.stringify(labels, null, 2));

    let datasetSize = null;
    if (Array.isArray(dataset)) {
      fs.writeFileSync(datasetPath, JSON.stringify(dataset));
      datasetSize = dataset.length;
    }

    if (metrics && typeof metrics === 'object') {
      const metricsPath = path.join(modelFolder, 'metrics.json');
      const toWrite = {
        accuracy:    metrics.accuracy ?? null,
        loss:        metrics.loss     ?? null,
        valAccuracy: metrics.valAccuracy ?? null,
        valLoss:     metrics.valLoss     ?? null,
        normalized:  metrics.normalized === true,
        datasetSize: metrics.datasetSize ?? datasetSize,
        epochs:      metrics.epochs ?? null,
        baseModel:   metrics.baseModel ?? null,
      };
      fs.writeFileSync(metricsPath, JSON.stringify(toWrite, null, 2));
    }

    res.status(200).json({
      message: 'Model saved successfully.',
      path: `dev/model-${timestamp}`,
      datasetSize,
    });
  } catch (err) {
    console.error('Save model error:', err);
    res.status(500).json({ error: 'Failed to save model.' });
  }
});


app.post('/api/promote-model', async (req, res) => {
  try {
    const { source } = req.body;
    const sourcePath = path.join(MODELS_DIR, source);
    const destPath = path.join(PROD_DIR, 'model-latest');

    if (!fs.existsSync(sourcePath)) {
      return res.status(404).json({ error: 'Source model not found.' });
    }

    // ลบของเดิม (model-latest)
    await fs.remove(destPath);
    // คัดลอกโฟลเดอร์ใหม่
    await fs.copy(sourcePath, destPath);

    res.json({ message: 'Model promoted to prod/model-latest.' });
  } catch (err) {
    console.error('Promote error:', err);
    res.status(500).json({ error: 'Failed to promote model.' });
  }
});

app.get('/api/validate-model/:modelFolder', async (req, res) => {
    try {
      const folder = req.params.modelFolder
      const modelPath = path.join(DEV_DIR, folder, 'model.json')
      const weightsPath = path.join(DEV_DIR, folder, 'group1-shard1of1.bin')
      const labelsPath = path.join(DEV_DIR, folder, 'labels.json')
  
      if (!fs.existsSync(modelPath)) return res.status(404).json({ error: 'ไม่พบ model.json' })
      if (!fs.existsSync(weightsPath)) return res.status(404).json({ error: 'ไม่พบ weights bin' })
      if (!fs.existsSync(labelsPath)) return res.status(404).json({ error: 'ไม่พบ labels.json' })
  
      const modelJson = JSON.parse(fs.readFileSync(modelPath, 'utf-8'))
      if (!modelJson.modelTopology || !modelJson.weightsManifest) {
        return res.status(400).json({ error: 'model.json ไม่สมบูรณ์' })
      }
  
      const labels = JSON.parse(fs.readFileSync(labelsPath, 'utf-8'))
      if (!Array.isArray(labels) || labels.length === 0) {
        return res.status(400).json({ error: 'labels.json ไม่ถูกต้องหรือว่างเปล่า' })
      }
  
      return res.status(200).json({ message: '✅ Model valid', labelsCount: labels.length })
    } catch (err) {
      console.error('Validate error:', err)
      return res.status(500).json({ error: 'ไม่สามารถตรวจสอบ model ได้' })
    }
  })

app.listen(PORT, () => {
  console.log(`🚀 Express server running at http://localhost:${PORT}`)
})
