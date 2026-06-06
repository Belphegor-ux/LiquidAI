"""One-off validation: preprocess ONLY chb01 and report label/channel stats.

Confirms the parser fix yields real preictal segments on actual EEG, and
surfaces per-file channel-count variation. Writes to a scratch dir so the
eventual full run (config.OUTPUT_DIR) is untouched.
"""

import json
from pathlib import Path

import numpy as np

import config
import preprocess

SCRATCH = Path("data/processed_chb01")


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    manifest = preprocess.parse_chbmit_manifest()
    chb01 = {"chb01": manifest.get("chb01", {})}
    n_seiz = sum(len(v) for v in chb01["chb01"].values())
    print(f"chb01 files in manifest: {len(chb01['chb01'])} | seizures annotated: {n_seiz}\n")

    all_seg, all_lab, pids = [], [], []
    for edf_file, seizure_times in chb01["chb01"].items():
        edf_path = config.DATA_DIR / "chb01" / edf_file
        if not edf_path.exists():
            edf_path = config.DATA_DIR / edf_file
        if not edf_path.exists():
            print(f"  - {edf_file}: not downloaded yet")
            continue
        try:
            data, sfreq = preprocess.preprocess_edf(edf_path)
            segments, labels = preprocess.label_segments(data, seizure_times, sfreq)
        except Exception as exc:  # noqa: BLE001
            print(f"  x {edf_file}: {exc}")
            continue
        n_ch = data.shape[0]
        tag = "" if n_ch == config.N_CHANNELS else f"  <-- {n_ch} ch != {config.N_CHANNELS}"
        print(
            f"  {edf_file}: {segments.shape[0]:>2} seg, "
            f"{int(labels.sum())} preictal, raw_ch={n_ch}{tag}"
        )
        if segments.shape[0] and n_ch == config.N_CHANNELS:
            all_seg.append(segments)
            all_lab.append(labels)
            pids.extend(["chb01"] * len(labels))

    if not all_seg:
        print("\nNo usable segments produced.")
        return
    seg = np.concatenate(all_seg)
    lab = np.concatenate(all_lab)
    np.save(SCRATCH / "segments.npy", seg)
    np.save(SCRATCH / "labels.npy", lab)
    (SCRATCH / "patient_ids.json").write_text(json.dumps(pids))
    print(
        f"\nTOTAL: {seg.shape[0]} segments shape {seg.shape[1:]} | "
        f"preictal {int(lab.sum())} | interictal {int((lab == 0).sum())}"
    )
    print(f"Saved to {SCRATCH}")


if __name__ == "__main__":
    main()
