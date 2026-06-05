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
DATA_DIR = _env_path("NEUROFLOW_DATA_DIR", "~/chbmit")          # raw CHB-MIT EDFs
OUTPUT_DIR = Path("data/processed")                             # preprocessed arrays
MODEL_DIR = Path("models")
LOG_DIR = Path("logs")
RESULTS_DIR = Path("results")

# --- Signal preprocessing -------------------------------------------------
SAMPLING_RATE = _env_int("NEUROFLOW_SR", 256)                  # CHB-MIT native (Hz)
FREQ_BAND = (0.5, 45.0)                                        # band-pass (Hz)
NOTCH_FREQ = 60.0                                             # US line noise (Hz)
N_CHANNELS = _env_int("NEUROFLOW_CHANNELS", 23)               # CHB-MIT scalp montage

# Each labelled example covers a 30-minute window.
WINDOW_MINUTES = 30
WINDOW_SECONDS = WINDOW_MINUTES * 60
PREICTAL_SECONDS = WINDOW_SECONDS                             # 30 min before onset = preictal

# Feeding 460,800 raw samples (30 min @ 256 Hz) through an RNN per example is
# infeasible in memory and compute. We decimate each window to DOWNSAMPLE_HZ,
# yielding a tractable sequence length while preserving the labelling scheme.
DOWNSAMPLE_HZ = _env_int("NEUROFLOW_DOWNSAMPLE_HZ", 2)        # effective rate after decimation
SEQ_LEN = WINDOW_SECONDS * DOWNSAMPLE_HZ                      # timesteps per example (3600 @ 2 Hz)

# --- Model ----------------------------------------------------------------
HIDDEN_SIZE = _env_int("NEUROFLOW_HIDDEN", 100)              # CfC-100
LEARNING_RATE = _env_float("NEUROFLOW_LR", 1e-3)
MAX_EPOCHS = _env_int("NEUROFLOW_EPOCHS", 150)
BATCH_SIZE = _env_int("NEUROFLOW_BATCH", 32)

# --- Split ----------------------------------------------------------------
RANDOM_SEED = _env_int("NEUROFLOW_SEED", 42)
TEST_FRACTION = _env_float("NEUROFLOW_TEST_FRAC", 0.30)      # of patients held out (val + test)
