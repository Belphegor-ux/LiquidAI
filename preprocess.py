"""Load CHB-MIT EDFs, filter, resample to 128 Hz, cut into short epochs, and label.

Outputs (in config.OUTPUT_DIR):
  segments.npy        (n_segments, SEQ_LEN, N_CHANNELS) float32  -- SEQ_LEN-sample epochs
  labels.npy          (n_segments,) int8   -- 1 preictal, 0 interictal
  patient_ids.json    [patient_id per segment]
  seizure_groups.json [seizure id per segment; "" for interictal] -- for per-seizure eval

Run:  python preprocess.py
"""

import json
import re
import shutil
import tempfile
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mne
import numpy as np

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

    # Select the canonical 23-channel montage by name, in a fixed order. CHB-MIT
    # recordings pad with placeholder "-" channels and extra references and vary
    # their channel ordering; picking an explicit allowlist drops the extras and
    # forces identical channels in identical order across every file. Check before
    # filtering so files missing canonical channels skip cheaply.
    canonical = list(config.CANONICAL_CHANNELS)
    missing = [c for c in canonical if c not in raw.ch_names]
    if missing:
        raise ValueError(f"missing {len(missing)} canonical channels, e.g. {missing[:3]}")
    raw.pick(canonical)
    raw.reorder_channels(canonical)  # pick() keeps source order; enforce ours

    raw.filter(config.FREQ_BAND[0], config.FREQ_BAND[1], verbose=False)
    raw.notch_filter(config.NOTCH_FREQ, verbose=False)
    # Resample to EPOCH_SAMPLE_RATE (128 Hz) -- high enough to keep the gamma band
    # (Nyquist 64 Hz) that carries preictal signal. MNE handles anti-aliasing.
    if round(raw.info["sfreq"]) != config.EPOCH_SAMPLE_RATE:
        raw.resample(config.EPOCH_SAMPLE_RATE, verbose=False)

    data = raw.get_data()  # (n_channels, n_samples) at EPOCH_SAMPLE_RATE
    if data.shape[0] != config.N_CHANNELS:
        raise ValueError(f"{data.shape[0]} channels after cleanup, expected {config.N_CHANNELS}")

    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True) + 1e-8
    data = (data - mean) / std
    return data, config.EPOCH_SAMPLE_RATE


def _epoch_starts(lo_sec, hi_sec):
    """Start times (seconds) of EPOCH_SECONDS epochs fully inside [lo, hi).

    Stepped by EPOCH_STRIDE_SECONDS; each epoch must end at or before hi.
    """
    starts, s = [], lo_sec
    while s + config.EPOCH_SECONDS <= hi_sec:
        starts.append(s)
        s += config.EPOCH_STRIDE_SECONDS
    return starts


def _overlaps_seizure(start_sec, end_sec, seizure_times):
    """True if [start_sec, end_sec) overlaps any seizure's exclusion zone."""
    for onset, offset in seizure_times:
        ex_start = onset - config.PREICTAL_HORIZON_SECONDS
        ex_end = offset + config.POSTICTAL_SECONDS
        if start_sec < ex_end and end_sec > ex_start:
            return True
    return False


def _slice_epoch(data, start_sec, sfreq, epoch_samples):
    """Return one (SEQ_LEN, n_channels) epoch starting at start_sec, or None."""
    i0 = int(round(start_sec * sfreq))
    i1 = i0 + epoch_samples
    if i1 > data.shape[1]:
        return None
    return data[:, i0:i1].T.astype(np.float32)


def label_segments(data, seizure_times, sfreq, rng):
    """Build preictal (1) and interictal (0) epoch examples from one recording.

    Preictal: EPOCH_SECONDS epochs inside [onset - PREICTAL_HORIZON_SECONDS,
    onset - SPH_SECONDS] for each seizure (SPH excludes the last minutes for
    real lead-time). Each preictal epoch is tagged with its seizure onset so the
    evaluator can score per-seizure.

    Interictal: epochs outside every seizure's [onset - HORIZON, offset +
    POSTICTAL] zone, randomly subsampled to INTERICTAL_PER_FILE to bound size.

    Returns (segments, labels, groups): segments (n, SEQ_LEN, N_CHANNELS);
    groups holds the seizure onset (int) for preictal epochs, -1 for interictal.
    """
    epoch_samples = int(config.EPOCH_SECONDS * sfreq)
    total_sec = data.shape[1] / sfreq

    segments, labels, groups = [], [], []

    # Preictal epochs, grouped by seizure onset.
    for onset, _offset in seizure_times:
        lo = max(0.0, onset - config.PREICTAL_HORIZON_SECONDS)
        hi = min(onset - config.SPH_SECONDS, total_sec)
        for start_sec in _epoch_starts(lo, hi):
            epoch = _slice_epoch(data, start_sec, sfreq, epoch_samples)
            if epoch is None:
                continue
            segments.append(epoch)
            labels.append(1)
            groups.append(int(onset))

    # Interictal candidates, then subsample.
    candidates = [
        s
        for s in _epoch_starts(0.0, total_sec)
        if not _overlaps_seizure(s, s + config.EPOCH_SECONDS, seizure_times)
    ]
    if len(candidates) > config.INTERICTAL_PER_FILE:
        keep = rng.choice(len(candidates), config.INTERICTAL_PER_FILE, replace=False)
        candidates = [candidates[i] for i in sorted(keep)]
    for start_sec in candidates:
        epoch = _slice_epoch(data, start_sec, sfreq, epoch_samples)
        if epoch is None:
            continue
        segments.append(epoch)
        labels.append(0)
        groups.append(-1)

    if not segments:
        return (
            np.empty((0, config.SEQ_LEN, config.N_CHANNELS), np.float32),
            np.empty((0,), np.int8),
            np.empty((0,), np.int64),
        )
    return (
        np.stack(segments).astype(np.float32),
        np.array(labels, dtype=np.int8),
        np.array(groups, dtype=np.int64),
    )


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


def _process_job(job):
    """Worker: load + label one EDF. Runs in a separate process.

    job = (patient_id, edf_path_str, seizure_times).
    Returns (patient_id, filename, segments|None, labels|None, groups|None, error|None).
    The interictal subsample is seeded deterministically per file (reproducible).
    """
    patient_id, edf_path, seizure_times = job
    name = Path(edf_path).name
    seed = (config.RANDOM_SEED + zlib.crc32(f"{patient_id}/{name}".encode())) % (2**32)
    rng = np.random.default_rng(seed)
    try:
        data, sfreq = preprocess_edf(Path(edf_path))
        segments, labels, groups = label_segments(data, seizure_times, sfreq, rng)
    except Exception as exc:  # noqa: BLE001 - keep going on bad files
        return patient_id, name, None, None, None, str(exc)
    return patient_id, name, segments, labels, groups, None


def _build_jobs(manifest):
    """Resolve EDF paths and return the list of worker jobs (skipping missing)."""
    jobs = []
    for patient_id in sorted(manifest):
        for edf_file, seizure_times in manifest[patient_id].items():
            edf_path = config.DATA_DIR / patient_id / edf_file
            if not edf_path.exists():
                edf_path = config.DATA_DIR / edf_file  # flat layout fallback
            if not edf_path.exists():
                print(f"  - {patient_id}/{edf_file} not found (skipped)")
                continue
            jobs.append((patient_id, str(edf_path), seizure_times))
    return jobs


def main():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Parsing CHB-MIT manifest...")
    manifest = parse_chbmit_manifest()
    if not manifest:
        raise SystemExit(
            f"No chb*_summary.txt files found under {config.DATA_DIR}. "
            "Download CHB-MIT first (see README)."
        )

    jobs = _build_jobs(manifest)
    print(f"Processing {len(jobs)} files with {config.NUM_WORKERS} worker(s)...")

    results = []
    temp_dir = Path(tempfile.mkdtemp(prefix="neuroflow_preprocess_"))

    try:
        with ProcessPoolExecutor(max_workers=config.NUM_WORKERS) as pool:
            for patient_id, name, segments, labels, groups, err in pool.map(_process_job, jobs):
                if err:
                    print(f"  x {patient_id}/{name}: {err}")
                    continue
                if segments is None or segments.shape[0] == 0:
                    continue

                # Write intermediate chunks to disk to save memory
                seg_file = temp_dir / f"{patient_id}_{name}_seg.npy"
                lab_file = temp_dir / f"{patient_id}_{name}_lab.npy"
                grp_file = temp_dir / f"{patient_id}_{name}_grp.npy"
                np.save(seg_file, segments)
                np.save(lab_file, labels)
                np.save(grp_file, groups)

                results.append(
                    {
                        "patient_id": patient_id,
                        "name": name,
                        "seg_file": seg_file,
                        "lab_file": lab_file,
                        "grp_file": grp_file,
                        "n_segments": len(labels),
                        "n_preictal": int(labels.sum()),
                    }
                )
                print(f"  {patient_id}/{name}: {len(labels)} seg ({int(labels.sum())} preictal)")

        if not results:
            raise SystemExit("No segments produced. Check the dataset path and summaries.")

        results.sort(key=lambda r: (r["patient_id"], r["name"]))  # stable patient/file order

        total_segments = sum(r["n_segments"] for r in results)
        print(f"\nMerging {total_segments} segments to disk...")

        segments_out = np.lib.format.open_memmap(
            config.OUTPUT_DIR / "segments.npy",
            mode="w+",
            dtype=np.float32,
            shape=(total_segments, config.SEQ_LEN, config.N_CHANNELS),
        )
        labels_out = np.lib.format.open_memmap(
            config.OUTPUT_DIR / "labels.npy", mode="w+", dtype=np.int8, shape=(total_segments,)
        )

        patient_ids = []
        seizure_groups = []  # globally-unique seizure id per segment; "" for interictal
        idx = 0
        for r in results:
            n = r["n_segments"]

            # Read chunks, write to memmap, then free memory
            seg_chunk = np.load(r["seg_file"])
            lab_chunk = np.load(r["lab_file"])
            grp_chunk = np.load(r["grp_file"])

            segments_out[idx : idx + n] = seg_chunk
            labels_out[idx : idx + n] = lab_chunk

            # Delete chunk files to save disk space
            r["seg_file"].unlink()
            r["lab_file"].unlink()
            r["grp_file"].unlink()

            patient_ids.extend([r["patient_id"]] * n)
            # Local group is the seizure onset (preictal) or -1 (interictal); make
            # it globally unique with patient + file so seizures never collide.
            for g in grp_chunk:
                seizure_groups.append("" if g < 0 else f"{r['patient_id']}|{r['name']}|{int(g)}")
            idx += n

        segments_out.flush()
        labels_out.flush()

        with open(config.OUTPUT_DIR / "patient_ids.json", "w") as f:
            json.dump(patient_ids, f)
        with open(config.OUTPUT_DIR / "seizure_groups.json", "w") as f:
            json.dump(seizure_groups, f)

        total_preictal = sum(r["n_preictal"] for r in results)
        total_interictal = total_segments - total_preictal
        print(f"Saved {total_segments} segments of shape {segments_out.shape[1:]}")
        print(f"Preictal: {total_preictal} | Interictal: {total_interictal}")

    finally:
        # Cleanup temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
