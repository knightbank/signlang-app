# _tfs_dataset/ — Data prep + training workspace

Off-app workspace for everything that produces a deployable model. Not part of the
running web app — these are the scripts and snapshots used to *build* what gets
deployed to `prod/model-latest`.

## 🌳 Branch & deployment layout (read me first if you just pulled the repo)

This repo is split into three independently versioned pieces:

| Repo | Location | Default branch |
|---|---|---|
| **Parent** (this repo `signlang-app`) | repo root | `main` |
| **Frontend** (`signlang-frontend.git`) | `frontend/` | `main` ⇄ `feat/exp-sequence-gru` |
| **Backend** (`signlang-backend.git`) | `backend/` | `main` ⇄ `feat/exp-sequence-gru` |

Both **frontend** and **backend** keep two branches:

- **`main`** — production-grade code only. Renders `/predict` + `/train` and serves
  `/api/*`. Every commit on `main` is deployable on its own. Used for hotfixes that
  must ship without dragging experimental code along.
- **`feat/exp-sequence-gru`** — the experimental track. A strict **superset** of
  `main` (main is merged in regularly), plus the `/exp/*` routes and `/api/exp/*`
  endpoints for the GRU sequence pipeline. This is **what actually gets built and
  deployed to the droplet** because it carries both production and experimental
  surfaces in one bundle.

The parent repo's submodule pointers track whichever sub-repo commit is currently
checked out. After a fresh clone:

```bash
git clone https://github.com/knightbank/signlang-app.git
cd signlang-app/frontend && git checkout feat/exp-sequence-gru && git pull
cd ../backend && git checkout feat/exp-sequence-gru && git pull
```

### What URLs map to what branch

| URL | Code source | Notes |
|---|---|---|
| https://139-59-57-147.nip.io/predict | frontend `feat/exp-sequence-gru` | production code (identical to `main`) |
| https://139-59-57-147.nip.io/train | frontend `feat/exp-sequence-gru` | production code (identical to `main`) |
| https://139-59-57-147.nip.io/exp/predict | frontend `feat/exp-sequence-gru` | exp-only routes |
| https://139-59-57-147.nip.io/exp/train | frontend `feat/exp-sequence-gru` | exp-only stub (Step 3 TODO) |
| https://139-59-57-147.nip.io/api/models | backend `feat/exp-sequence-gru` | production endpoint |
| https://139-59-57-147.nip.io/api/exp/models | backend `feat/exp-sequence-gru` | exp-only endpoint |

The deployed model artifacts (Dense MLP for `/predict`, GRU for `/exp/predict`) are
NOT in git — they live on the droplet under `/root/SIGNLANG-APP/models/...` and are
mirrored locally under `backups/` in this folder.

### Working-with-branches workflow

1. **Hotfix to production**: branch off `main`, commit, push `main`, then
   `git checkout feat/exp-sequence-gru && git merge main` so the exp track picks up
   the fix automatically.
2. **Experimental work**: stay on `feat/exp-sequence-gru`. Only files under
   `frontend/src/components/exp/`, `backend/index.js` (`/api/exp/*` block), and
   `train_node/train_gru.js` are exp-only. Touching `main`-tracked files there will
   create merge conflicts later — keep prod changes on `main`.
3. **Deploy** always builds from `feat/exp-sequence-gru` because of the superset
   property. The exp routes are kept clearly demarcated by a yellow nav banner
   when the user is on `/exp/*` so end-users don't confuse the two surfaces.

## Folder layout

```
_tfs_dataset/
├── download_*.py             # one-shot data fetchers (Mendeley/ASL-HG)
├── extract_*.py              # MediaPipe landmark extraction (Python, signlang_env)
├── extracted*.json           # the actual training datasets (versioned v2, v3, ...)
├── train_node/               # Node + TF.js training scripts
│   ├── train.js              # 8-class baseline (Mendeley บ + ASL-HG via Thai↔ASL map)
│   ├── train_v2.js           # 9-class (adds ร from ASL R)
│   ├── train_v3.js           # 10-class (adds จ from ASL J — negative-result for paper)
│   ├── train_gru.js          # GRU sequence model on pseudo-sequences (exp track)
│   └── package.json          # @tensorflow/tfjs only
├── backups/                  # deployed-model snapshots + rollback.sh + restore guide
├── bor/                      # Mendeley One-Stage-TFS, 48 verified Thai บ images (CC BY 4.0)
├── handshape_inventory.json  # TSL 44-letter handshape spec (Gemini-aided, verify needed)
├── thai_fs_chart.jpg         # PMC Silanon 2017 Figure 2 — 16 one-stroke reference
└── asl_chart.pdf             # Standard ASL alphabet (NCO Inc) — for Thai↔ASL mapping
```

The big stuff is **gitignored** — `asl_unzip/`, `ASL_Processed_Images.zip` (366 MB),
`train_node/node_modules/`, and intermediate `trained_model_*/` build outputs. Run
`download_asl.py` to refetch the ASL-HG zip, then `extract_*.py` to rebuild
`extracted_v*.json`. Final shipped model artifacts live under `backups/`.

## Environment

Use **only** `signlang_env`. NEVER `pip install` into `deepquant_env` or any other env.
See `CLAUDE.md` Rule 1 at the repo root.

```bash
conda create -y -n signlang_env python=3.11
conda run -n signlang_env pip install mediapipe==0.10.18 opencv-python pillow numpy requests
```

For Node training:

```bash
cd train_node
npm install
node train_v3.js   # or train_v2.js / train_gru.js / train.js
```

## Typical pipeline (to reproduce or extend)

1. `python download_asl.py` — fetch Mendeley ASL-HG zip
2. `unzip ASL_Processed_Images.zip -d asl_unzip` — expand 36k cropped JPGs
3. `python extract_all.py` — run MediaPipe on Mendeley บ + 8 mapped ASL letters → `extracted.json`
4. `python extract_add_ror.py` — add ร → `extracted_v2.json`
5. `python extract_add_jor.py` — add จ → `extracted_v3.json`
6. `node train_node/train_v3.js` — train Dense MLP (10-class) → `trained_model_v3/`
7. Deploy via the pattern in `backups/README.md` + atomic swap on the droplet
8. Mirror new snapshot into `backups/model-<date>-<descriptor>/` and update `backups/README.md`

## Rollback

`backups/rollback.sh` is installed on the droplet as `/root/rollback.sh`. See
`backups/ROLLBACK_GUIDE.md` for the operator runbook.

## Adding a new letter

Use `extract_add_jor.py` as a template:
1. Add the source folder loop (e.g., `asl_unzip/asl_processed/{train,test}/<LETTER>/`)
2. Load the previous `extracted_v{N-1}.json` as baseline
3. Append new samples → `extracted_v{N}.json`
4. Copy `train_node/train_v3.js` → `train_v{N}.js`, point `DATASET_PATH` and `OUT_DIR` to the new version
5. Run, deploy, archive

If the new class **replaces** an existing one (e.g., the planned ม update from expert
ground truth), filter the old class out of the baseline before appending — see the
plan section at the bottom of `C:\Users\knigh\.claude\plans\elegant-conjuring-umbrella.md`.
