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
