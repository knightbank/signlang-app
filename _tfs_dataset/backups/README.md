# Model Backups

Snapshots of every notable production / experimental model. Each folder is a
self-contained TFJS Layers model and can be uploaded straight back into
`backend/models/{prod,exp/prod}/model-latest` on the droplet to restore that
checkpoint.

## How to restore one of these to the droplet

```bash
# Example: restore the Dense baseline
scp -r model-2026-06-04-prod-dense-8class-valacc9959 \
    root@139.59.57.147:/root/SIGNLANG-APP/models/prod/model-latest-restored
ssh root@139.59.57.147
  cd /root/SIGNLANG-APP/models/prod
  mv model-latest model-latest-old.$(date +%s)   # keep the live one just in case
  mv model-latest-restored model-latest
  docker restart signlang-app_backend_1
```

For the exp model use `/root/SIGNLANG-APP/models/exp/prod/` instead.

## What is in each backup

### `model-2026-06-04-prod-dense-8class-valacc9959/`
- Live `prod/model-latest` as of 2026-06-04
- **Architecture**: Multi-Layer Perceptron — Dense(63→256→128→64→8) + BatchNorm + Dropout(0.3)
- **Input**: single 63-D normalized landmark vector (per frame)
- **Classes** (`labels.json`): `["น", "บ", "ม", "ย", "ล", "ว", "ส", "อ"]`
- **Validation accuracy**: 99.59%
- **Dataset**: 2048 normalized landmarks from Mendeley One-Stage-TFS (Thai บ, 48) + Mendeley ASL-HG (mapped via Thai↔ASL handshape correspondence for the other 7 classes, 250 each)
- **Trained on**: 2026-05-27 via `_tfs_dataset/train_node/train.js`
- **Served at**: https://139-59-57-147.nip.io/predict
- **Files**: model.json (3.5 KB) + group1-shard1of1.bin (231 KB) + labels.json + metrics.json + dataset.json (replay buffer for incremental training)
- **SHA256**:
  - model.json: `344b9e58f04144986a80da2784fb8b787a89d61cf2657bb6ad025f0a960ed8c4`
  - group1-shard1of1.bin: `35d47eb7dd982bcc0d8ec701f093cf24a4ffba8f9c2fda2e4cd701c1354fd8bb`

### `model-2026-06-04-exp-gru-8class-valacc9976/`
- Live `exp/prod/model-latest` as of 2026-06-04
- **Architecture**: 2-layer GRU(64) + Dropout(0.3) + Dense(32, ReLU) + Dense(8, softmax)
- **Input**: 15-frame rolling window of 63-D normalized landmarks → shape [batch, 15, 63]
- **Classes** (`labels.json`): namespaced — `["consonant_น", "consonant_บ", "consonant_ม", "consonant_ย", "consonant_ล", "consonant_ว", "consonant_ส", "consonant_อ"]`
- **Validation accuracy**: 99.76%
- **Bootstrap method**: pseudo-sequences (repeat the same static 63-D vector 15 times with σ=0.005 Gaussian noise per frame), so the same Mendeley + ASL-HG dataset trains a sequence model
- **Trained on**: 2026-05-30 via `_tfs_dataset/train_node/train_gru.js`
- **Served at**: https://139-59-57-147.nip.io/exp/predict
- **Files**: model.json (3.4 KB) + group1-shard1of1.bin (202 KB) + labels.json + metrics.json + dataset.json
- **SHA256**:
  - model.json: `5e8cd69a17f9ab0bdbfa57f9ac73ea9b3531263c70324c81786fc1c52068c7d2`
  - group1-shard1of1.bin: `525438a2a836064c1efd6c9e75a8f98cdfb0765985dcbb46cb9aeb9f3f36bb79`

## Where else this is backed up

Same artifacts also live on the droplet under:
- `/root/SIGNLANG-APP/models/archive/model-2026-06-04-prod-dense-8class-valacc9959/`
- `/root/SIGNLANG-APP/models/exp/archive/model-2026-06-04-exp-gru-8class-valacc9976/`

Source data for both models: `_tfs_dataset/extracted.json` (2048 normalized landmarks). Source images that produced the landmarks live under `_tfs_dataset/bor/` (Mendeley One-Stage-TFS, 48 images of บ) and `_tfs_dataset/asl_unzip/asl_processed/{train,test}/{A,B,L,M,N,S,W,Y}/` (ASL-HG sample).
