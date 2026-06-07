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

## ⚠️ Repo is mid-pivot (read before running)

The repo is currently in a **work-in-progress spectral pivot** and is **not
runnable end-to-end as-is**:

- `model.py` was changed to `input_size=115` (hardcoded — should move to
  `config.py`) and `train.py` loads `segments_spectral.npy` (produced by
  `process_spectral.py` + `extract_features.py`). This is the band-power
  feature path ("path b").
- The committed checkpoint `models/cfc100-epoch=02-val_loss=0.1097.ckpt`
  expects the **old** `(1280, 23)` raw-epoch input, and `server.py` loads that
  checkpoint via `find_checkpoint()`. So the **model code and the saved
  checkpoint / API are inconsistent** right now.

To run the *original* working pipeline, revert `model.py` (`input_size` back to
`config.N_CHANNELS`) and `train.py` (load `segments.npy`). To continue the
spectral pivot, run `process_spectral.py` to build `segments_spectral.npy`, then
retrain — and update `server.py` / the frontend to the new input contract.

See `RESULTS.md` for the honest metrics (mean per-patient test AUROC ≈ 0.475)
and `CLAUDE.md` for full project status.
