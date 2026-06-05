"""Load CHB-MIT EDFs, filter, decimate, segment into 30-min windows, and label.

Outputs (in config.OUTPUT_DIR):
  segments.npy     (n_segments, SEQ_LEN, N_CHANNELS) float32
  labels.npy       (n_segments,) int8  -- 1 preictal, 0 interictal
  patient_ids.json [patient_id per segment]

Run:  python preprocess.py
"""

import json

import mne
import numpy as np
from scipy.signal import resample_poly
from tqdm import tqdm

import config


def preprocess_edf(edf_path):
    """Load one EDF, band-pass + notch filter, z-normalise per channel.

    Returns (data, sfreq) where data is (n_channels, n_samples).
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    raw.filter(config.FREQ_BAND[0], config.FREQ_BAND[1], verbose=False)
    raw.notch_filter(config.NOTCH_FREQ, verbose=False)

    data = raw.get_data()  # (n_channels, n_samples)
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


def label_segments(data, seizure_times, sfreq):
    """Split a recording into 30-min windows; label preictal vs interictal.

    A window is preictal if it falls entirely within the 30 minutes before a
    seizure onset. Returns (segments, labels).
    """
    window_samples = int(config.WINDOW_SECONDS * sfreq)
    n_segments = data.shape[1] // window_samples

    segments, labels = [], []
    for seg_idx in range(n_segments):
        start = seg_idx * window_samples
        end = start + window_samples
        window = data[:, start:end]

        start_sec = start / sfreq
        end_sec = end / sfreq

        is_preictal = False
        for seiz_start, _seiz_end in seizure_times:
            preictal_start = seiz_start - config.PREICTAL_SECONDS
            if preictal_start <= start_sec < seiz_start and end_sec <= seiz_start:
                is_preictal = True
                break

        segments.append(_decimate_window(window, sfreq))
        labels.append(1 if is_preictal else 0)

    if not segments:
        return np.empty((0, config.SEQ_LEN, config.N_CHANNELS), np.float32), np.empty((0,), np.int8)
    return np.stack(segments).astype(np.float32), np.array(labels, dtype=np.int8)


def parse_chbmit_manifest():
    """Parse chb*_summary.txt files into {patient: {edf_file: [(start, end), ...]}}."""
    manifest = {}
    for summary_file in sorted(config.DATA_DIR.glob("**/chb*_summary.txt")):
        patient_id = summary_file.stem.replace("_summary", "")
        manifest[patient_id] = {}
        current_file = None
        with open(summary_file) as f:
            for line in f:
                if "File Name:" in line or "File:" in line:
                    current_file = line.split(":", 1)[1].strip()
                    manifest[patient_id][current_file] = []
                elif "Seizure" in line and "Time:" in line and current_file:
                    times = line.split("Time:")[1].strip().split("-")
                    try:
                        start = int("".join(c for c in times[0] if c.isdigit()))
                        end = int("".join(c for c in times[1] if c.isdigit()))
                        manifest[patient_id][current_file].append((start, end))
                    except (ValueError, IndexError):
                        continue
    return manifest


def main():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Parsing CHB-MIT manifest...")
    manifest = parse_chbmit_manifest()
    if not manifest:
        raise SystemExit(f"No chb*_summary.txt files found under {config.DATA_DIR}. "
                         "Download CHB-MIT first (see README).")

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
