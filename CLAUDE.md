# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

**NeuroFlow** — cross-patient epileptic seizure *prediction* (not detection) from
scalp EEG. Classifies a 30-minute window as **preictal** (0–30 min before a
seizure) vs **interictal**, using a CfC-100 liquid neural network trained on the
CHB-MIT dataset with a patient-level split (no leakage). Targets: ≥80%
sensitivity, <0.3 false alarms/hour, edge-deployable inference.

## Layout (flat, by design — scripts import each other directly)

| File | Role |
|------|------|
| `config.py` | All constants; env-overridable. **Change values here, never inline.** |
| `preprocess.py` | EDF → filter → decimate → 30-min windows → labels |
| `split_patients.py` | Patient-level train/val/test split |
| `model.py` | `CfCSeizurePredictor` (PyTorch Lightning) |
| `train.py` | Training loop, class-balanced sampler, AUC checkpoint |
| `evaluate.py` | Test metrics + ROC; `find_checkpoint()` reused by demo/benchmark |
| `demo.py` | Streamlit UI |
| `benchmark.py` | CPU latency (edge proxy) |

## Run order

`preprocess.py → split_patients.py → train.py → evaluate.py` (then `demo.py` / `benchmark.py`).

## Conventions

- **No hard-coded values.** Add tunables to `config.py` with an env override.
- **Shapes:** preprocessed segments are `(N, SEQ_LEN, N_CHANNELS)` — i.e.
  `(batch, time, channels)`, fed directly to the CfC. Do not reintroduce a
  per-timestep Python loop in `forward`.
- **Numeric stability:** logits + `BCEWithLogitsLoss`, `gradient_clip_val=1.0`,
  AUROC guarded against single-class batches.
- **No patient leakage:** any new split logic must keep patients disjoint across
  train/val/test; `split_patients.py` asserts this.
- Keep files focused (<400 lines), prefer pure functions, handle bad EDFs
  gracefully (skip + log, don't crash the run).

## Data

CHB-MIT EDFs live outside the repo (default `~/chbmit`, set `NEUROFLOW_DATA_DIR`).
`data/`, `models/`, `logs/`, `results/`, `ncps/` are git-ignored.

## Dependencies

`ncps` (Liquid networks), `torch`, `pytorch-lightning`, `mne`, `scikit-learn`,
`scipy`, `streamlit`. See `requirements.txt`.

## Current status (updated 2026-06-06)

**Run 3 shipped.** 19 patients (chb01–chb19), 28,193 epochs, 133 seizures. Best
checkpoint `models/cfc100-epoch=02-val_loss=0.1097.ckpt`.

- **Honest headline metric: mean per-patient (subject-wise) test AUROC ≈ 0.475
  (chance).** The *pooled* AUROC (0.578) is inflated by differing per-patient
  preictal base rates — always report the per-patient number (`evaluate.py`
  prints both). Cross-patient prediction is the open problem; see `RESULTS.md`.
- Pipeline, checkpoint, and LFM2.5 explainer are all functional end-to-end.
- A silent montage bug was fixed: `config.CANONICAL_CHANNELS` selects the 23-name
  bipolar montage by name (chb12+ pad with placeholder/extra/duplicate channels);
  this also enforces consistent channel order.

**Resume next session (needs Antigravity API-contract coordination):** to get real
signal, choose (b) spectral/band-power features [changes the `(1280, 23)` model
input → breaks the current API contract] or (c) patient-specific calibration /
fine-tuning [keeps the contract, reframes as patient-adaptive]. Consider
leave-one-patient-out CV + Wilson CIs (3-patient test set is high-variance).
`frontend/` (Vite/React) + `server.py` (FastAPI) are Antigravity's; both are now
committed (run `npm install` in `frontend/` on a fresh checkout — `node_modules`
is git-ignored). Scaffolding for both improvement paths is in the repo:
`extract_features.py` (path b, band-power) and `finetune.py` (path c, per-patient).

**Env gotchas:** Windows + Bash tool — no PowerShell here-strings in Bash (use
`-m`×N or `<<'EOF'`); set `PYTHONUTF8=1` for scripts (cp932 console); repo is in
OneDrive (avoid git worktrees there); training logs to TensorBoard event files,
not `metrics.csv`. Use the `/commit` skill for safe staging/commits.
