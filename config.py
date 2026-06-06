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

# Some CHB-MIT recordings append non-EEG channels mid-session (e.g. chb04 has
# ECG, chb09 has VNS), giving 24+ channels. Drop these so every file presents
# the same 23-channel montage; files that still mismatch are skipped, not fatal.
NON_EEG_CHANNELS = frozenset({"ECG", "EKG", "VNS", "LOC-ROC", "ROC-LOC", "-", "--"})

# Each labelled example covers a 30-minute window.
WINDOW_MINUTES = 30
WINDOW_SECONDS = WINDOW_MINUTES * 60
PREICTAL_SECONDS = WINDOW_SECONDS  # 30 min before onset = preictal

# A preictal example is the slice [onset - PREICTAL_SECONDS, onset). When a
# seizure starts early in a recording the full 30 min isn't available (the rest
# would lie in the previous file); require at least this much real pre-onset
# data, then pad to SEQ_LEN. Seizures with less are skipped.
MIN_PREICTAL_SECONDS = _env_int("NEUROFLOW_MIN_PREICTAL_SEC", 900)  # >= half the window

# Interictal windows must stay clear of every seizure's influence: exclude any
# window overlapping [onset - PREICTAL_SECONDS, offset + POSTICTAL_SECONDS].
POSTICTAL_SECONDS = _env_int("NEUROFLOW_POSTICTAL_SEC", 1800)

# Feeding 460,800 raw samples (30 min @ 256 Hz) through an RNN per example is
# infeasible in memory and compute. We decimate each window to DOWNSAMPLE_HZ,
# yielding a tractable sequence length while preserving the labelling scheme.
DOWNSAMPLE_HZ = _env_int("NEUROFLOW_DOWNSAMPLE_HZ", 2)  # effective rate after decimation
SEQ_LEN = WINDOW_SECONDS * DOWNSAMPLE_HZ  # timesteps per example (3600 @ 2 Hz)

# --- Model ----------------------------------------------------------------
HIDDEN_SIZE = _env_int("NEUROFLOW_HIDDEN", 100)  # CfC-100
LEARNING_RATE = _env_float("NEUROFLOW_LR", 1e-3)
MAX_EPOCHS = _env_int("NEUROFLOW_EPOCHS", 150)
BATCH_SIZE = _env_int("NEUROFLOW_BATCH", 32)

# --- Split ----------------------------------------------------------------
RANDOM_SEED = _env_int("NEUROFLOW_SEED", 42)
TEST_FRACTION = _env_float("NEUROFLOW_TEST_FRAC", 0.30)  # of patients held out (val + test)

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
