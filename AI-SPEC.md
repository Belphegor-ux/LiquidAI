# AI-SPEC: NeuroFlow Seizure Prediction Model

Design contract for the AI component. Keep this in sync with `config.py` and `model.py`.

## 1. Task definition

| | |
|---|---|
| **Type** | Binary sequence classification (preictal vs interictal) |
| **Input** | 30-min EEG window, decimated to `(SEQ_LEN, N_CHANNELS)` = `(3600, 23)` @ 2 Hz |
| **Output** | Single preictal probability ∈ [0, 1] |
| **Positive class** | Preictal = the 30 min immediately before a labelled seizure onset |
| **Decision** | Alarm when `p > 0.5` (threshold tunable on the validation ROC) |

## 2. Model

- **Backbone:** `ncps.torch.CfC` (Closed-form Continuous-time), `HIDDEN_SIZE=100`.
- **Head:** Linear(100→64) → ReLU → Dropout(0.2) → Linear(64→1) → logit.
- **Loss:** `BCEWithLogitsLoss` (stable under 16-bit mixed precision).
- **Optimiser:** Adam, `lr=1e-3`, `gradient_clip_val=1.0`.
- **Why CfC:** continuous-time dynamics suit streaming EEG; small + fast for edge.

## 3. Evaluation strategy

- **Split:** patient-level 70/15/15 (train/val/test), **disjoint patients** — the
  headline metric is generalisation to *unseen people*, asserted in code.
- **Primary metric:** sensitivity (recall on preictal). **Guardrails:** AUROC,
  specificity, false-alarms/hour (<0.3 target).
- **Imbalance:** preictal is rare → `WeightedRandomSampler` in training;
  AUROC/sensitivity (not accuracy) are the real signals.
- **Checkpointing:** best `val_auc`; early stop patience 20.

## 4. Critical failure modes & mitigations

| Failure mode | Mitigation |
|---|---|
| Patient leakage inflates metrics | Patient-level split + disjointness assertion |
| Model predicts all-interictal (imbalance) | Class-balanced sampler; track sensitivity/AUC |
| CfC NaN divergence | Gradient clipping; logits loss; lower `lr` fallback |
| GPU OOM (RTX 3050, 12 GB) | Decimated sequences; 16-bit precision; reduce `BATCH_SIZE` |
| Decimation discards seizure-relevant high-freq content | Documented trade-off; next step = per-band feature extraction |
| Single-class validation batch breaks AUROC | Guarded in `on_validation_epoch_end` |

## 5. Acceptance criteria

- **Minimum:** trains end-to-end, patient-disjoint split, sensitivity ≥ 75%, demo runs.
- **Target:** sensitivity ≥ 80%, AUC ≥ 0.83, CPU latency validated.

## 6. Production / monitoring (future)

Per-patient calibration, drift monitoring on incoming EEG distribution, alarm-rate
dashboards, and a human-in-the-loop confirmation step before any intervention.
