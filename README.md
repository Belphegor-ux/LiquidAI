# NeuroFlow: Cross-Patient Seizure Prediction with Liquid Neural Networks

**48-hour hackathon entry** · CHB-MIT dataset · CfC-100 · target 80% sensitivity, <0.3 FP/h

Predict epileptic seizures **30 minutes before onset** from scalp EEG, using a
Closed-form Continuous-time (CfC) liquid neural network that generalises across
patients it has never seen.

## Pipeline

```
preprocess.py      EDF -> filter -> decimate -> 30-min windows -> labels   (data/processed/*.npy)
split_patients.py  patient-level train/val/test split (no leakage)         (split.json)
model.py           CfC-100 LightningModule
train.py           train with class-balanced sampling + AUC checkpointing  (models/*.ckpt)
evaluate.py        sensitivity / specificity / AUC on held-out patients    (results/*)
demo.py            Streamlit prediction UI
benchmark.py       CPU inference latency (edge proxy)
config.py          all tunable constants (env-overridable)
```

## Quick start

```bash
# 1. Liquid network library + deps
git clone https://github.com/mlech26l/ncps.git && pip install -e ./ncps
pip install -r requirements.txt

# 2. CHB-MIT data (~42 GB) into ~/chbmit (override with NEUROFLOW_DATA_DIR)
aws s3 sync --no-sign-request s3://physionet-open/chbmit/1.0.0/ ~/chbmit/

# 3. Run the pipeline
python preprocess.py
python split_patients.py
python train.py            # or: python train.py > logs/training.log 2>&1 &
python evaluate.py
streamlit run demo.py
```

## Notes on this scaffold (deviations from the original roadmap)

The roadmap's reference code had three issues that would prevent it from running;
they are fixed here:

1. **Infeasible sequence length.** A raw 30-min window is 460,800 samples/channel.
   Storing `(n_segments, 23, 460800)` and stepping it through an RNN in a Python
   loop is impossible in memory and compute. `preprocess.py` now decimates each
   window to `DOWNSAMPLE_HZ` (default 2 Hz -> `SEQ_LEN = 3600`), and `model.py`
   uses the optimized `ncps.torch.CfC` module instead of a manual per-step loop.
   *Trade-off:* low-rate decimation loses high-frequency detail; per-band feature
   extraction is the recommended next step for accuracy.
2. **Missing import / numeric stability.** `os` is imported where used; the model
   uses `BCEWithLogitsLoss` (stable under 16-bit mixed precision) instead of
   `Sigmoid + BCELoss`, plus `gradient_clip_val=1.0` to prevent CfC divergence.
3. **Class imbalance & batched eval.** Training uses a `WeightedRandomSampler` so
   the rare preictal class is learned; evaluation runs in batches to avoid OOM.

All constants live in [`config.py`](config.py) and can be overridden via
environment variables (e.g. `NEUROFLOW_DOWNSAMPLE_HZ=4 python preprocess.py`).

## References

- Hasani et al. (2022), *Liquid Time-Constant Networks*, Nature Machine Intelligence.
- Lechner & Hasani, [`ncps`](https://github.com/mlech26l/ncps) — CfC / LTC implementations.
- CHB-MIT Scalp EEG Database — https://physionet.org/content/chbmit/
