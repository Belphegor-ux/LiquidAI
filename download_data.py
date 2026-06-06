"""Download CHB-MIT scalp-EEG recordings from PhysioNet into the data dir.

Resumable: existing complete files are skipped; partial files resume via curl -C -.
The per-patient summary (chbNN-summary.txt) is fetched first and parsed for the
exact EDF list, so we never guess filenames.

Usage:
  python download_data.py                 # default patients chb01..chb10
  python download_data.py chb01 chb05      # explicit patients
  NEUROFLOW_PATIENTS=chb01,chb02 python download_data.py
"""

import os
import subprocess
import sys

import config

# AWS Open Data mirror -- ~30x faster than physionet.org over HTTPS.
BASE_URL = "https://physionet-open.s3.amazonaws.com/chbmit/1.0.0"
DEFAULT_PATIENTS = [f"chb{n:02d}" for n in range(1, 11)]


def _patients_from_args() -> list[str]:
    if len(sys.argv) > 1:
        return sys.argv[1:]
    env = os.environ.get("NEUROFLOW_PATIENTS")
    if env:
        return [p.strip() for p in env.split(",") if p.strip()]
    return DEFAULT_PATIENTS


def _curl(url: str, dest) -> bool:
    """Download url to dest with resume; return True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["curl", "-fsSL", "-C", "-", "-o", str(dest), url],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"    ! failed ({result.returncode}): {result.stderr.strip()[:200]}")
        return False
    return True


def _parse_edf_names(summary_path) -> list[str]:
    names = []
    for line in summary_path.read_text(errors="ignore").splitlines():
        if line.lstrip().startswith("File Name:"):
            names.append(line.split(":", 1)[1].strip())
    return names


def main() -> None:
    patients = _patients_from_args()
    print(f"Target dir: {config.DATA_DIR}")
    print(f"Patients: {', '.join(patients)}\n")

    total_files = 0
    for patient in patients:
        pdir = config.DATA_DIR / patient
        summary = pdir / f"{patient}-summary.txt"
        print(f"{patient}:")

        if not summary.exists():
            if not _curl(f"{BASE_URL}/{patient}/{patient}-summary.txt", summary):
                print("  x summary download failed; skipping patient")
                continue
        edf_names = _parse_edf_names(summary)
        print(f"  summary lists {len(edf_names)} EDF files")

        for name in edf_names:
            dest = pdir / name
            if dest.exists() and dest.stat().st_size > 0:
                continue
            print(f"  -> {name}", flush=True)
            if _curl(f"{BASE_URL}/{patient}/{name}", dest):
                total_files += 1

    print(f"\nDone. Downloaded {total_files} new EDF file(s) into {config.DATA_DIR}")


if __name__ == "__main__":
    main()
