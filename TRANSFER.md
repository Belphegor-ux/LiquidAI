# TRANSFER.md — Moving NeuroFlow to another machine

`git clone` carries **all code** (including `frontend/`), but **not** the data,
model weights, or `node_modules` — those are git-ignored and must be copied or
rebuilt manually. Follow the steps below to get a fully working checkout.

## 1. Copy these manually (git does NOT carry them)

| What | Path | Approx size | Notes |
|------|------|------------|-------|
| Raw CHB-MIT EDFs | `~/chbmit` (outside repo; Windows: `C:\Users\<you>\chbmit`) | **36 GB** | `config.DATA_DIR`, override with `NEUROFLOW_DATA_DIR` |
| Preprocessed arrays | `<repo>/data/` (incl. `data/processed/*.npy`) | **3.1 GB** | `config.OUTPUT_DIR = data/processed` |
| Checkpoints + LFM weights | `<repo>/models/` | **2.2 GB** | incl. `models/lfm2.5-1.2b-instruct` (2.34 GB) and `*.ckpt` |
| Training logs | `<repo>/logs/`, `<repo>/lightning_logs/` | ~tiny | optional; regenerated on retrain |
| Results | `<repo>/results/` | ~tiny | optional |

Use an external drive or a network copy — these are too large for git.
On the new machine, place `data/`, `models/`, `logs/`, `results/` **inside the
repo root**, and `chbmit/` wherever you point `NEUROFLOW_DATA_DIR` (default `~/chbmit`).

## 2. Set up the new machine

```bash
git clone <repo-url> "Liquid AI"
cd "Liquid AI"

# Python
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

# Frontend (rebuilds the 114 MB node_modules that git ignores)
cd frontend
npm install
cd ..
```

If `~/chbmit` is not where you put the raw data, point the env var at it:

```bash
# Windows (PowerShell)
$env:NEUROFLOW_DATA_DIR = "D:\chbmit"
# Always set this for scripts on a cp932 (Japanese) Windows console:
$env:PYTHONUTF8 = "1"
```

## 3. Verify

```bash
python -c "import config; print(config.DATA_DIR, config.OUTPUT_DIR)"
# Already-preprocessed? jump straight to:
python evaluate.py
# Otherwise rebuild from raw:  preprocess.py -> split_patients.py -> train.py -> evaluate.py
```

## 4. Run the dashboard (backend + frontend)

```bash
# Terminal 1 — FastAPI inference API on http://localhost:8000
$env:PYTHONUTF8 = "1"     # Windows cp932 consoles
python server.py

# Terminal 2 — Vite dev server on http://localhost:5173
cd frontend
npm run dev
```

The sidebar pulls real per-patient risk from `/api/ward` (batched model
inference over the replayed epochs) and the spatial panel from `/api/predict`.
If the backend is down, the frontend falls back to placeholders / a
deterministic feed.

## Status (read before running)

The repo is in the **runnable raw-epoch state** (restored in commit
`840d42b`): `model.py` uses `input_size=config.N_CHANNELS`, `train.py` loads
`segments.npy`, and `server.py` pins the matching raw-epoch checkpoint via
`_pin_raw_checkpoint()`. The pipeline runs end-to-end as-is.

Optional spectral pivot ("path b", band-power features) is **scaffolding only**
and not wired into the default pipeline: `extract_features.py` /
`process_spectral.py` build `segments_spectral.npy`, but using them means
retraining with a 115-input model and updating the `server.py` / frontend input
contract. `finetune.py` is the alternative per-patient path ("path c").

See `RESULTS.md` for the honest metrics (mean per-patient test AUROC ≈ 0.475)
and `CLAUDE.md` for full project status.
