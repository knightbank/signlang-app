# signlang-app — Thai Sign Language Web App

Real-time Thai fingerspelling recognition that runs in any modern phone browser. The user signs in front of the camera, MediaPipe Hands extracts 21 hand keypoints, and a small TensorFlow.js classifier predicts the Thai consonant.

🌐 **Live**: <https://139-59-57-147.nip.io/predict>

| Surface | URL | Model | Status |
|---|---|---|---|
| Production prediction | [/predict](https://139-59-57-147.nip.io/predict) | Dense MLP — 10 classes (อ บ ล ม น ส ว ย ร จ) | val_acc 99.93% |
| Production training UI | [/train](https://139-59-57-147.nip.io/train) | — | volunteer-friendly incremental data collection |
| Experimental prediction | [/exp/predict](https://139-59-57-147.nip.io/exp/predict) | GRU sequence model — 15-frame rolling window | val_acc 99.76% on pseudo-sequences |
| Experimental training (stub) | [/exp/train](https://139-59-57-147.nip.io/exp/train) | — | placeholder; will host video-clip recording |

## Quick start

```bash
git clone https://github.com/knightbank/signlang-app.git
cd signlang-app

# Backend (port 3001)
cd backend
npm install
node index.js

# Frontend (port 5173) — in another shell
cd ../frontend
npm install
npm run dev
```

The dev server proxies `/api/*` and `/models/*` to the backend automatically (see [frontend/vite.config.js](frontend/vite.config.js)).

## Repo layout

```
signlang-app/                ← single monorepo (was 3 nested repos until 2026-06-29)
├── README.md                ← you are here
├── frontend/                ← Vue 3 + Vite app (TF.js + MediaPipe Hands)
│   └── src/
│       ├── components/
│       │   ├── PredictPage.vue          ← /predict
│       │   ├── TrainingPage.vue         ← /train
│       │   └── exp/                     ← /exp/* (only on feat/exp-sequence-gru)
│       ├── composables/useHandLandmarks.js
│       └── constants/thaiConsonants.js  ← THAI_CONSONANTS list
├── backend/                 ← Node/Express server
│   └── index.js             ← /api/models, /api/save-model, /api/promote-model
│                              + /api/exp/* mirror (only on feat/exp-sequence-gru)
├── nginx/                   ← reverse proxy config used in production Docker
└── _tfs_dataset/            ← data prep + training scripts + model snapshots
    ├── README.md            ← workspace + branch + reproduction guide
    ├── reference.html       ← per-class training summary (open in browser)
    ├── extract_*.py         ← MediaPipe landmark extraction (signlang_env)
    ├── train_node/          ← TFJS training scripts (train.js, train_v2.js, …)
    └── backups/             ← deployed-model snapshots + rollback.sh + ROLLBACK_GUIDE.md
```

## Branches

Two branches, both pushed to GitHub:

- **`main`** — production-only stack. The `/predict` + `/train` pages and the `/api/*` endpoints. Every commit on this branch is independently deployable.
- **`feat/exp-sequence-gru`** — deployed superset. `main` content plus the experimental `/exp/*` routes, `/api/exp/*` endpoints, and the GRU sequence pipeline. This is what actually runs on the droplet.

```bash
# Production-only snapshot
git checkout main

# Full deployed code (what the live URL serves)
git checkout feat/exp-sequence-gru
```

Production hotfixes branch from `main`, get committed + pushed to `main`, then merged forward into `feat/exp-sequence-gru`. Experimental work stays on the feature branch.

## ML pipeline (one diagram)

```
camera frame  →  MediaPipe Hands  →  21 keypoints (x, y, z)
              →  Hand-relative normalize (translate to wrist, scale by wrist→middle-MCP)
              →  63-D feature vector
              →  [Dense MLP / GRU sequence model]
              →  Thai consonant probability + skeleton overlay
```

Same normalization step on both training and inference paths — see [`frontend/src/composables/useHandLandmarks.js`](frontend/src/composables/useHandLandmarks.js).

## Reference documents

- 📊 [_tfs_dataset/reference.html](_tfs_dataset/reference.html) — per-class training summary with thumbnails (open locally)
- 📦 [_tfs_dataset/README.md](_tfs_dataset/README.md) — data workspace, reproduction pipeline, branch & URL routing
- 🔧 [_tfs_dataset/backups/ROLLBACK_GUIDE.md](_tfs_dataset/backups/ROLLBACK_GUIDE.md) — one-button rollback runbook for the droplet
- 🗂 [_tfs_dataset/handshape_inventory.json](_tfs_dataset/handshape_inventory.json) — 44-letter TSL handshape spec (partial)

## Deployment

Production runs in two Docker containers on DigitalOcean droplet `139.59.57.147`:

- `signlang-app_backend_1` — Express server, port 3001 internal
- `signlang-app_frontend_nginx_1` — Nginx serving `frontend/dist` + reverse proxy

The model artifacts (TF.js Layers format) live on the droplet under `/root/SIGNLANG-APP/models/` and are NOT git-tracked. Their snapshots are mirrored to `_tfs_dataset/backups/` in this repo for safety. To roll back: `ssh root@139.59.57.147 && /root/rollback.sh prod`.

## License & attribution

- This repo: project owner ([knightbank](https://github.com/knightbank))
- Training data:
  - Thai บ images — [Mendeley One-Stage-TFS](https://data.mendeley.com/datasets/rknd3wbz42/1) (CC BY 4.0)
  - ASL handshape images — [Mendeley ASL-HG](https://data.mendeley.com/datasets/j4y5w2c8w9/1) (CC BY 4.0) — used because some Thai handshapes are visually identical to ASL letters
- Reference charts — [Silanon 2017 PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5611514/) (Thai), NCO Inc (ASL)
- Pre-trained model used for keypoint extraction: [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html) by Google

## History

This was originally three separate git repositories — `signlang-app` (parent with nginx config and notes), `signlang-frontend` (Vue app), and `signlang-backend` (Express server) — held together by nested `.git` directories without `.gitmodules`. On 2026-06-29 the layout was consolidated into a single monorepo. The legacy repos `github.com/knightbank/signlang-frontend` and `github.com/knightbank/signlang-backend` remain available as historical archives; all future commits land in this one. Tagged `pre-monorepo-2026-06-29` on all three repos for traceability.
