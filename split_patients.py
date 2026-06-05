"""Patient-level train/val/test split (no patient leakage between sets).

Reads config.OUTPUT_DIR/{segments,labels,patient_ids}; writes split.json.

Run:  python split_patients.py
"""

import json

import numpy as np
from sklearn.model_selection import train_test_split

import config


def build_split(patient_ids):
    """Return index lists keyed by split, with patients disjoint across sets."""
    unique_patients = sorted(set(patient_ids))
    print(f"Total patients: {len(unique_patients)}")

    train_patients, holdout = train_test_split(
        unique_patients, test_size=config.TEST_FRACTION, random_state=config.RANDOM_SEED
    )
    val_patients, test_patients = train_test_split(
        holdout, test_size=0.5, random_state=config.RANDOM_SEED
    )

    train_set, val_set, test_set = set(train_patients), set(val_patients), set(test_patients)
    return {
        "train_idx": [i for i, p in enumerate(patient_ids) if p in train_set],
        "val_idx": [i for i, p in enumerate(patient_ids) if p in val_set],
        "test_idx": [i for i, p in enumerate(patient_ids) if p in test_set],
        "train_patients": sorted(train_set),
        "val_patients": sorted(val_set),
        "test_patients": sorted(test_set),
    }


def main():
    with open(config.OUTPUT_DIR / "patient_ids.json") as f:
        patient_ids = json.load(f)

    split = build_split(patient_ids)

    # Sanity: no patient appears in more than one set.
    sets = [set(split["train_patients"]), set(split["val_patients"]), set(split["test_patients"])]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2]), \
        "Patient leakage detected between splits"

    with open(config.OUTPUT_DIR / "split.json", "w") as f:
        json.dump(split, f)

    print(f"Patients  -> train {len(split['train_patients'])} | "
          f"val {len(split['val_patients'])} | test {len(split['test_patients'])}")
    print(f"Segments  -> train {len(split['train_idx'])} | "
          f"val {len(split['val_idx'])} | test {len(split['test_idx'])}")

    labels = np.load(config.OUTPUT_DIR / "labels.npy")
    for name in ("train", "val", "test"):
        idx = split[f"{name}_idx"]
        preictal = int(labels[idx].sum()) if idx else 0
        print(f"  {name}: {preictal} preictal / {len(idx)} total")


if __name__ == "__main__":
    main()
