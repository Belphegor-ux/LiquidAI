"""Load CHB-MIT EDFs, filter, decimate, segment into 30-min windows, and label.

Outputs (in config.OUTPUT_DIR):
  segments.npy     (n_segments, SEQ_LEN, N_CHANNELS) float32
  labels.npy       (n_segments,) int8  -- 1 preictal, 0 interictal
  patient_ids.json [patient_id per segment]

Run:  python preprocess.py
"""

import json
import re

import mne
import numpy as np
from scipy.signal import resample_poly

import config

# Matches CHB-MIT seizure annotation lines, e.g.
#   "Seizure Start Time: 2996 seconds"
#   "Seizure 1 End Time: 3036 seconds"   (numbered variant for multi-seizure files)
_SEIZURE_TIME_RE = re.compile(r"Seizure\s*\d*\s*(Start|End)\s*Time:\s*(\d+)", re.IGNORECASE)
# Legacy single-line variant: "Seizure 1 Time: 2996 - 3036"
_SEIZURE_RANGE_RE = re.compile(r"Seizure\s*\d*\s*Time:\s*(\d+)\s*-\s*(\d+)", re.IGNORECASE)


def preprocess_edf(edf_path):
    """Load one EDF, band-pass + notch filter, z-normalise per channel.

    Returns (data, sfreq) where data is (n_channels, n_samples).
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

    # Drop non-EEG channels (ECG/VNS/etc.) so every file has the same montage.
    drop = [c for c in raw.ch_names if c.upper() in config.NON_EEG_CHANNELS]
    if drop:
        raw.drop_channels(drop)

    raw.filter(config.FREQ_BAND[0], config.FREQ_BAND[1], verbose=False)
    raw.notch_filter(config.NOTCH_FREQ, verbose=False)

    data = raw.get_data()  # (n_channels, n_samples)
    if data.shape[0] != config.N_CHANNELS:
        raise ValueError(f"{data.shape[0]} channels after cleanup, expected {config.N_CHANNELS}")

    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True) + 1e-8
    data = (data - mean) / std
    return data, raw.info["sfreq"]


def _decimate_window(window, src_hz):
    """Resample a (n_channels, n_samples) window from src_hz to DOWNSAMPLE_HZ.

    Returns (SEQ_LEN, n_channels) so it feeds an RNN as (time, features).
    """
    decimated = resample_poly(window, up=config.DOWNSAMPLE_HZ, down=int(src_hz), axis=1)
    # Pad / trim to the exact expected length so every example is uniform.
    if decimated.shape[1] < config.SEQ_LEN:
        pad = config.SEQ_LEN - decimated.shape[1]
        decimated = np.pad(decimated, ((0, 0), (0, pad)), mode="edge")
    decimated = decimated[:, : config.SEQ_LEN]
    return decimated.T.astype(np.float32)  # (SEQ_LEN, n_channels)


def _overlaps_seizure(start_sec, end_sec, seizure_times):
    """True if [start_sec, end_sec) overlaps any seizure's exclusion zone."""
    for onset, offset in seizure_times:
        ex_start = onset - config.PREICTAL_SECONDS
        ex_end = offset + config.POSTICTAL_SECONDS
        if start_sec < ex_end and end_sec > ex_start:
            return True
    return False


def label_segments(data, seizure_times, sfreq):
    """Build preictal (1) and interictal (0) examples from one recording.

    Preictal: the slice [onset - PREICTAL_SECONDS, onset) for each seizure (one
    positive per seizure), provided at least MIN_PREICTAL_SECONDS of real
    pre-onset data exists; it is decimated and padded to SEQ_LEN.

    Interictal: non-overlapping WINDOW_SECONDS windows that do not overlap any
    seizure's [onset - PREICTAL_SECONDS, offset + POSTICTAL_SECONDS] zone, so
    negatives are clear of pre/post-ictal activity.

    Returns (segments, labels) with segments shaped (n, SEQ_LEN, N_CHANNELS).
    """
    window_samples = int(config.WINDOW_SECONDS * sfreq)
    min_preictal_samples = int(config.MIN_PREICTAL_SECONDS * sfreq)
    n_samples = data.shape[1]

    segments, labels = [], []

    # Preictal positives: explicit per-seizure slice ending at onset.
    for onset, _offset in seizure_times:
        onset_sample = int(onset * sfreq)
        start_sample = max(0, onset_sample - window_samples)
        window = data[:, start_sample:onset_sample]
        if window.shape[1] < min_preictal_samples:
            continue  # too little real pre-onset data (rest lies in prior file)
        segments.append(_decimate_window(window, sfreq))
        labels.append(1)

    # Interictal negatives: clean non-overlapping windows.
    for seg_idx in range(n_samples // window_samples):
        start = seg_idx * window_samples
        end = start + window_samples
        if _overlaps_seizure(start / sfreq, end / sfreq, seizure_times):
            continue
        segments.append(_decimate_window(data[:, start:end], sfreq))
        labels.append(0)

    if not segments:
        return np.empty((0, config.SEQ_LEN, config.N_CHANNELS), np.float32), np.empty((0,), np.int8)
    return np.stack(segments).astype(np.float32), np.array(labels, dtype=np.int8)


def parse_chbmit_manifest():
    """Parse chb*_summary.txt files into {patient: {edf_file: [(start, end), ...]}}."""
    manifest = {}
    # Real CHB-MIT files are named "chb01-summary.txt" (hyphen); the "*" between
    # "chb" and "summary" matches both the hyphen and the legacy underscore form.
    for summary_file in sorted(config.DATA_DIR.glob("**/chb*summary.txt")):
        patient_id = summary_file.stem.replace("-summary", "").replace("_summary", "")
        manifest[patient_id] = {}
        current_file = None
        pending_start = None  # holds a "Start Time" awaiting its "End Time"
        with open(summary_file) as f:
            for line in f:
                stripped = line.lstrip()
                # Anchor to line start so "Number of Seizures in File: 1" is not
                # mistaken for a file-name header.
                if stripped.startswith("File Name:") or stripped.startswith("File:"):
                    current_file = line.split(":", 1)[1].strip()
                    manifest[patient_id][current_file] = []
                    pending_start = None
                    continue
                if current_file is None:
                    continue

                # Legacy single-line range: "Seizure 1 Time: 2996 - 3036".
                range_match = _SEIZURE_RANGE_RE.search(line)
                if range_match:
                    start, end = int(range_match.group(1)), int(range_match.group(2))
                    manifest[patient_id][current_file].append((start, end))
                    continue

                # Standard two-line form: separate Start Time / End Time lines.
                time_match = _SEIZURE_TIME_RE.search(line)
                if not time_match:
                    continue
                kind, seconds = time_match.group(1).lower(), int(time_match.group(2))
                if kind == "start":
                    pending_start = seconds
                elif pending_start is not None:  # "end" with a matching start
                    manifest[patient_id][current_file].append((pending_start, seconds))
                    pending_start = None
    return manifest


def main():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Parsing CHB-MIT manifest...")
    manifest = parse_chbmit_manifest()
    if not manifest:
        raise SystemExit(
            f"No chb*_summary.txt files found under {config.DATA_DIR}. "
            "Download CHB-MIT first (see README)."
        )

    all_segments, all_labels, patient_ids = [], [], []

    for patient_id in sorted(manifest):
        print(f"\n{patient_id}:")
        for edf_file, seizure_times in manifest[patient_id].items():
            edf_path = config.DATA_DIR / patient_id / edf_file
            if not edf_path.exists():
                edf_path = config.DATA_DIR / edf_file  # flat layout fallback
            if not edf_path.exists():
                print(f"  - {edf_file} not found (download incomplete?)")
                continue

            try:
                data, sfreq = preprocess_edf(edf_path)
                segments, labels = label_segments(data, seizure_times, sfreq)
            except Exception as exc:  # noqa: BLE001 - keep going on bad files
                print(f"  x {edf_file}: {exc}")
                continue

            if segments.shape[0] == 0:
                continue
            all_segments.append(segments)
            all_labels.append(labels)
            patient_ids.extend([patient_id] * len(labels))
            print(f"    {edf_file}: {len(labels)} segments ({int(labels.sum())} preictal)")

    if not all_segments:
        raise SystemExit("No segments produced. Check the dataset path and summaries.")

    segments = np.concatenate(all_segments, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print("\nSaving preprocessed data...")
    np.save(config.OUTPUT_DIR / "segments.npy", segments)
    np.save(config.OUTPUT_DIR / "labels.npy", labels)
    with open(config.OUTPUT_DIR / "patient_ids.json", "w") as f:
        json.dump(patient_ids, f)

    print(f"Saved {segments.shape[0]} segments of shape {segments.shape[1:]}")
    print(f"Preictal: {int(labels.sum())} | Interictal: {int((labels == 0).sum())}")


if __name__ == "__main__":
    main()
