"""NeuroFlow inference API.

Serves real per-patient predictions from the CfC seizure predictor over the
preprocessed CHB-MIT epochs, plus real per-region band-power for the dashboard's
spatial-activity panel. Consistent with the runnable raw-epoch model: it loads
`segments.npy` (N, SEQ_LEN, 23) and the 23-channel checkpoint.
"""

import json
import re

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config
from evaluate import find_checkpoint
from extract_features import extract_band_power
from model import CfCSeizurePredictor

app = FastAPI(title="NeuroFlow API")

# Allow the React frontend (any localhost port) to call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CANONICAL 23-channel montage -> the 5 scalp regions shown in the UI.
# Indices are positions in config.CANONICAL_CHANNELS, grouped by electrode topography.
REGION_CHANNELS = {
    "Frontal (Fp1-Fp2)": (0, 4, 8, 12),  # FP1-F7, FP1-F3, FP2-F4, FP2-F8
    "Temporal (T3-T4)": (1, 2, 13, 14, 18, 19, 20, 21, 22),
    "Central (C3-C4)": (5, 6, 9, 10, 16, 17),
    "Parietal (P3-P4)": (7, 11),  # P3-O1, P4-O2
    "Occipital (O1-O2)": (3, 15),  # P7-O1, P8-O2
}

# --- lazy singletons (loaded once, reused across requests) -----------------
_model = None
_device = None
_segments = None
_patient_index: dict[str, np.ndarray] = {}
_cursor: dict[str, int] = {}  # per-patient replay position


def _pin_raw_checkpoint() -> str:
    """Pick the checkpoint that matches the raw model (input_size == N_CHANNELS).

    find_checkpoint() returns the newest by mtime, which can be a 115-input
    spectral checkpoint. We instead select the matching-architecture ckpt with
    the lowest val_loss in its filename, falling back to find_checkpoint().
    """
    matching = []
    for p in config.MODEL_DIR.glob("*.ckpt"):
        try:
            hp = torch.load(p, map_location="cpu", weights_only=False).get("hyper_parameters", {})
            if int(hp.get("input_size", -1)) == config.N_CHANNELS:
                matching.append(p)
        except Exception:
            continue
    if not matching:
        return find_checkpoint()

    def _val_loss(path):
        m = re.search(r"val_loss=(\d+\.\d+)", path.name)
        return float(m.group(1)) if m else float("inf")

    matching.sort(key=lambda p: (_val_loss(p), -p.stat().st_mtime))
    return str(matching[0])


def _load():
    """Load model + memory-mapped segments + per-patient epoch index once."""
    global _model, _device, _segments
    if _model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = _pin_raw_checkpoint()
        _model = CfCSeizurePredictor.load_from_checkpoint(ckpt, map_location=_device)
        _model.to(_device).eval()
    if _segments is None:
        _segments = np.load(config.OUTPUT_DIR / "segments.npy", mmap_mode="r")
    if not _patient_index:
        with open(config.OUTPUT_DIR / "patient_ids.json") as f:
            ids = json.load(f)
        grouped: dict[str, list] = {}
        for i, pid in enumerate(ids):
            grouped.setdefault(pid, []).append(i)
        _patient_index.update({k: np.asarray(v) for k, v in grouped.items()})
    return _model, _device, _segments


def _normalize_patient_id(pid: str) -> str:
    """Frontend sends 'CHB-01'; the processed data uses 'chb01'."""
    return pid.lower().replace("-", "").strip()


def _regional_activity(segment: np.ndarray, risk_pct: float) -> list[dict]:
    """Real per-region band power from one (SEQ_LEN, 23) epoch.

    Returns UI-ready rows. `activity` is the gamma-band spatial pattern (the
    preictal-relevant band) scaled by the model's risk; `variance` is the
    within-region spread of total power. All values come from the actual EEG.
    """
    data = np.asarray(segment, dtype=np.float64).T  # (23, SEQ_LEN)
    bands = extract_band_power(data, fs=config.EPOCH_SAMPLE_RATE)  # each (23,)
    gamma = np.asarray(bands["Gamma"], dtype=np.float64)
    total = sum(np.asarray(bands[b], dtype=np.float64) for b in bands)  # (23,)

    region_gamma, region_var = {}, {}
    for name, chans in REGION_CHANNELS.items():
        idx = list(chans)
        region_gamma[name] = float(np.mean(gamma[idx]))
        region_var[name] = float(np.std(total[idx])) if len(idx) > 1 else 0.0

    gmax = max(region_gamma.values()) + 1e-9
    vmax = max(region_var.values()) + 1e-9

    rows = []
    for name in REGION_CHANNELS:
        norm = region_gamma[name] / gmax  # real spatial pattern, 0..1
        nv = region_var[name] / vmax
        activity = float(np.clip(norm * (35.0 + 0.6 * risk_pct), 3.0, 100.0))
        variance = float(np.clip(nv * (12.0 + 0.3 * risk_pct), 1.0, 48.0))
        rows.append({"name": name, "activity": round(activity, 1), "variance": round(variance, 1)})
    return rows


@app.get("/api/predict")
def predict(patient_id: str = "CHB-01"):
    try:
        model, device, segments = _load()
        pid = _normalize_patient_id(patient_id)
        if pid not in _patient_index:
            raise HTTPException(status_code=404, detail=f"Unknown patient '{patient_id}'")

        epochs = _patient_index[pid]
        # Sequential replay of the patient's real epochs, advancing each poll.
        pos = _cursor.get(pid, 0) % len(epochs)
        _cursor[pid] = (pos + 1) % len(epochs)
        seg_idx = int(epochs[pos])

        sample = np.array(segments[seg_idx])  # (SEQ_LEN, 23); copy out of the read-only mmap
        x = torch.from_numpy(sample[None]).float().to(device)
        with torch.no_grad():
            prob = float(model.predict_proba(x).cpu().item())

        risk_pct = prob * 100.0
        return {
            "status": "success",
            "patient_id": pid,
            "segment_index": seg_idx,
            "predict_proba": prob,
            "risk_level": "HIGH" if prob > 0.5 else "LOW",
            "channel_data": _regional_activity(sample, risk_pct),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
