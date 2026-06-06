"""Central configuration for NeuroFlow.

All tunable constants live here so no values are hard-coded across scripts.
Override any of these with an environment variable of the same name.
"""

import os
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    return Path(os.path.expanduser(os.environ.get(name, default)))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


# --- Paths ----------------------------------------------------------------
DATA_DIR = _env_path("NEUROFLOW_DATA_DIR", "~/chbmit")  # raw CHB-MIT EDFs
OUTPUT_DIR = Path("data/processed")  # preprocessed arrays
MODEL_DIR = Path("models")
LOG_DIR = Path("logs")
RESULTS_DIR = Path("results")

# --- Signal preprocessing -------------------------------------------------
SAMPLING_RATE = _env_int("NEUROFLOW_SR", 256)  # CHB-MIT native (Hz)
FREQ_BAND = (0.5, 45.0)  # band-pass (Hz)
NOTCH_FREQ = 60.0  # US line noise (Hz)
N_CHANNELS = _env_int("NEUROFLOW_CHANNELS", 23)  # CHB-MIT scalp montage

# CHB-MIT canonical 23-channel bipolar montage. Recordings are inconsistent:
# chb12+ interleave placeholder "-" separators (MNE renames duplicates to "--0",
# "--1", ...), chb15 appends extra FC*-Ref/CP*-Ref channels, and the montage
# duplicates "T8-P8" (MNE -> "T8-P8-0" / "T8-P8-1"). Rather than drop a blocklist
# (fragile -- it left 8 patients at 28 channels and silently excluded them), we
# SELECT exactly these names in this order. This guarantees an identical montage
# AND identical channel ordering across every file; recordings missing any of
# these (a few short chb13 files) raise and are skipped, not fatal.
CANONICAL_CHANNELS = (
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8-0",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
    "P7-T7",
    "T7-FT9",
    "FT9-FT10",
    "FT10-T8",
    "T8-P8-1",
)

# --- Epoching (short-epoch design) ----------------------------------------
# We classify short epochs, not one 30-min window. At 128 Hz a 10 s epoch is 1280
# timesteps (a tractable CfC sequence) and KEEPS the gamma band that carries
# preictal signal -- the old 2 Hz decimation (Nyquist 1 Hz) discarded everything
# above 1 Hz. Short epochs also multiply the positive class ~100x, easing the
# 22:1 imbalance.
EPOCH_SECONDS = _env_int("NEUROFLOW_EPOCH_SEC", 10)
EPOCH_SAMPLE_RATE = _env_int("NEUROFLOW_EPOCH_HZ", 128)  # Nyquist 64 Hz keeps gamma
EPOCH_STRIDE_SECONDS = _env_int("NEUROFLOW_EPOCH_STRIDE_SEC", EPOCH_SECONDS)  # =len: no overlap
SEQ_LEN = EPOCH_SECONDS * EPOCH_SAMPLE_RATE  # 1280 timesteps per example

# --- Labeling -------------------------------------------------------------
# Preictal = epochs in [onset - PREICTAL_HORIZON_SECONDS, onset - SPH_SECONDS].
# SPH (seizure prediction horizon) drops the final minutes before onset so the
# model learns genuine lead-time prediction, not last-second detection.
PREICTAL_HORIZON_SECONDS = _env_int("NEUROFLOW_PREICTAL_HORIZON_SEC", 1800)  # 30 min
SPH_SECONDS = _env_int("NEUROFLOW_SPH_SEC", 300)  # 5 min lead-time, excluded
# Interictal = epochs outside [onset - PREICTAL_HORIZON, offset + POSTICTAL].
POSTICTAL_SECONDS = _env_int("NEUROFLOW_POSTICTAL_SEC", 1800)
# Interictal epochs vastly outnumber preictal; randomly keep at most this many
# per recording (seeded) to bound dataset size and the class imbalance.
INTERICTAL_PER_FILE = _env_int("NEUROFLOW_INTERICTAL_PER_FILE", 25)

# Preprocessing parallelism: EDF load + filter is CPU-bound and independent per
# file, so we fan out across processes. Capped to bound RAM (big files are ~700 MB
# each in memory). Set NEUROFLOW_WORKERS=1 to force serial.
NUM_WORKERS = _env_int("NEUROFLOW_WORKERS", max(1, min(6, (os.cpu_count() or 4) - 2)))

# --- Model ----------------------------------------------------------------
HIDDEN_SIZE = _env_int("NEUROFLOW_HIDDEN", 100)  # CfC-100
LEARNING_RATE = _env_float("NEUROFLOW_LR", 1e-3)
# Regularization to close the cross-patient train->test gap (overfitting).
WEIGHT_DECAY = _env_float("NEUROFLOW_WEIGHT_DECAY", 1e-2)  # AdamW decoupled decay
DROPOUT = _env_float("NEUROFLOW_DROPOUT", 0.4)
MAX_EPOCHS = _env_int("NEUROFLOW_EPOCHS", 150)
BATCH_SIZE = _env_int("NEUROFLOW_BATCH", 32)
# Early-stopping patience on val_loss; set NEUROFLOW_PATIENCE=0 to disable entirely
# (train the full MAX_EPOCHS -- useful when the val set is small/noisy).
EARLY_STOP_PATIENCE = _env_int("NEUROFLOW_PATIENCE", 20)
# The model uses focal loss for imbalance and preprocessing caps interictal epochs
# per file, so the weighted sampler is OFF by default -- stacking all three over-
# predicts positives and collapses specificity. Enable: NEUROFLOW_WEIGHTED_SAMPLER=1.
USE_WEIGHTED_SAMPLER = _env_int("NEUROFLOW_WEIGHTED_SAMPLER", 0) == 1

# --- Split ----------------------------------------------------------------
RANDOM_SEED = _env_int("NEUROFLOW_SEED", 42)
TEST_FRACTION = _env_float("NEUROFLOW_TEST_FRAC", 0.30)  # of patients held out (val + test)

# --- Evaluation -----------------------------------------------------------
# Clinical target: keep false alarms under this rate. The decision threshold is
# tuned on the validation set to the lowest value (highest sensitivity) whose
# val FA/h <= target, then frozen for the test set. Override the tuned value with
# NEUROFLOW_THRESHOLD to force a fixed threshold.
EVAL_TARGET_FA_PER_H = _env_float("NEUROFLOW_TARGET_FA_PER_H", 0.3)
_threshold_env = os.environ.get("NEUROFLOW_THRESHOLD")
EVAL_THRESHOLD = float(_threshold_env) if _threshold_env else None

# --- LFM explanation layer (optional) -------------------------------------
# Turns a CfC preictal probability into a natural-language clinical alert using a
# Liquid LFM2 model run in-process via transformers (on-device, no server, no API
# key -- the model is public). Default is the 1.2B instruct model, which fits a
# ~3 GB GPU; override NEUROFLOW_LLM_DEVICE=cpu to keep the GPU free for training.
# Local dir populated by download_lfm.py (HF's Xet CDN throttles, so we fetch via
# parallel range curl). Falls back to the HF repo id if the local copy is absent.
_LOCAL_LFM = Path("models/lfm2.5-1.2b-instruct")
LLM_MODEL = os.environ.get(
    "NEUROFLOW_LLM_MODEL",
    str(_LOCAL_LFM) if _LOCAL_LFM.exists() else "LiquidAI/LFM2.5-1.2B-Instruct",
)
LLM_DEVICE = os.environ.get("NEUROFLOW_LLM_DEVICE", "auto")  # auto | cuda | cpu
LLM_TEMPERATURE = _env_float("NEUROFLOW_LLM_TEMP", 0.3)
LLM_MAX_TOKENS = _env_int("NEUROFLOW_LLM_MAX_TOKENS", 300)
