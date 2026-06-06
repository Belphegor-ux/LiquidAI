# NeuroFlow — Results (run 3, 2026-06-06)

Cross-patient epileptic **seizure prediction** (preictal vs interictal) from scalp
EEG, using a CfC-100 liquid neural network on CHB-MIT with a patient-level split.

## TL;DR (read this before presenting)

The **engineering pipeline is complete and works end-to-end**; the **model does not
yet achieve real cross-patient predictive power**. Report the *per-patient* metric,
not the pooled one — the pooled number is misleading on this dataset.

- **Mean per-patient test AUROC: 0.475 (≈ chance).** This is the honest headline.
- Pooled test AUROC 0.578 looks better but is **inflated** by differing per-patient
  preictal base rates (see "Methodology note").

## Dataset

- **19 patients** (chb01–chb19), **28,193** 10 s epochs @ 128 Hz, **133** seizures.
- Patient-level split, no leakage: 13 train / 3 val (chb02,09,12) / 3 test (chb01,06,17).
- Per-channel z-norm; canonical 23-channel bipolar montage selected by name.

## Test metrics (held-out patients chb01/06/17, best checkpoint, epoch 2)

| Metric | Value | Target |
|---|---|---|
| **Mean per-patient AUROC** | **0.475** | — (honest metric) |
| Pooled AUROC | 0.578 | — (inflated, see below) |
| Pooled AUPRC | 0.617 | baseline 0.56 |
| Per-seizure sensitivity | 0.300 (6/20) | ≥ 0.80 |
| False alarms / hour | 2.79 | < 0.30 |
| Per-window sensitivity / specificity | 0.011 / 0.992 | — |

Per-patient AUROC: chb01 0.507 · chb06 0.512 · chb17 0.406.

## Methodology note — why pooled AUROC is misleading here

Test patients have very different preictal fractions (e.g. chb06 73% preictal vs
chb01 44%). Pooling them lets a classifier score well by separating *patients*
rather than *preictal vs interictal within a patient*. A deployed predictor runs
inside one patient, so the clinically correct frame is **subject-wise AUROC**
(`evaluate.py` reports both; trust the per-patient one).

## What works (engineering)

- EDF → filter → 128 Hz → 10 s epochs → preictal/interictal labels (SPH lead-time).
- Montage handling across all 19 patients (placeholder/duplicate channels resolved).
- Patient-safe split with leakage assertion.
- CfC-100 training (focal loss, AdamW weight-decay, dropout, attention pooling),
  batch-256 throughput tuning (~8× speedup on the laptop GPU).
- Evaluation: per-seizure sensitivity, event-wise FA/h, AUROC/AUPRC, val-tuned
  threshold, **per-patient** breakdown, ROC plot.
- LFM2.5-1.2B on-device explainer turning a probability into a clinical alert.

## The open problem & honest next steps

Raw time-domain epochs + a small CfC don't capture *transferable* preictal signal.
Standard remedies (not attempted before the deadline):
1. **Spectral / band-power features** (the literature's biggest lever). Changes the
   model input, so it requires updating the inference/API contract.
2. **Patient-specific calibration / fine-tuning** — adapt to a little of each
   patient's own data; commonly reaches 0.8+ but reframes the claim as
   "patient-adaptive" rather than zero-shot cross-patient.
3. More patients + leave-one-patient-out CV with confidence intervals (current
   3-patient test set is high-variance).

Artifacts: `results/evaluation.json`, `results/roc_curve.png`,
`models/cfc100-epoch=02-val_loss=0.1097.ckpt`.
