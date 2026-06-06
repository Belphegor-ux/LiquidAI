"""Evaluate the trained CfC model on the held-out test patients.

Writes results/evaluation.json and results/roc_curve.png.

Run:  python evaluate.py [path/to/checkpoint.ckpt]
"""

import json
import sys

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset

import config
from model import CfCSeizurePredictor


def find_checkpoint(explicit=None):
    """Return the best checkpoint path: explicit arg, else best val_auc, else last."""
    if explicit:
        return explicit
    ckpts = list(config.MODEL_DIR.glob("cfc100-*.ckpt"))
    if ckpts:
        # Filename encodes val_auc; pick the highest.
        return str(
            max(
                ckpts,
                key=lambda p: float(p.stem.split("val_auc=")[-1]) if "val_auc=" in p.stem else -1.0,
            )
        )
    last = config.MODEL_DIR / "last.ckpt"
    if last.exists():
        return str(last)
    raise SystemExit(f"No checkpoint found in {config.MODEL_DIR}. Train first.")


def predict_in_batches(model, segments, device):
    loader = DataLoader(TensorDataset(segments), batch_size=config.BATCH_SIZE)
    preds = []
    for (batch,) in loader:
        preds.append(model.predict_proba(batch.to(device)).cpu())
    return torch.cat(preds).numpy()


def main():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = find_checkpoint(sys.argv[1] if len(sys.argv) > 1 else None)
    print("Loading", ckpt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CfCSeizurePredictor.load_from_checkpoint(ckpt, map_location=device).to(device)
    model.eval()

    segments = torch.from_numpy(np.load(config.OUTPUT_DIR / "segments.npy")).float()
    labels = np.load(config.OUTPUT_DIR / "labels.npy")
    with open(config.OUTPUT_DIR / "split.json") as f:
        test_idx = json.load(f)["test_idx"]

    test_segments = segments[test_idx]
    test_labels = labels[test_idx]
    test_preds = predict_in_batches(model, test_segments, device)

    # AUROC/ROC are undefined when the test set has a single class (which can
    # happen if the held-out test patient has no preictal segments).
    single_class = len(np.unique(test_labels)) < 2
    if single_class:
        print("WARNING: test set has a single class; AUC/ROC undefined.")
        auc = float("nan")
    else:
        auc = roc_auc_score(test_labels, test_preds)
    cm = confusion_matrix(test_labels, (test_preds > 0.5).astype(int), labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    print(f"Test AUC:     {auc:.3f}")
    print(f"Sensitivity:  {sensitivity:.3f}")
    print(f"Specificity:  {specificity:.3f}")
    print(f"Confusion matrix [[tn fp][fn tp]]:\n{cm}")

    with open(config.RESULTS_DIR / "evaluation.json", "w") as f:
        json.dump(
            {
                "checkpoint": ckpt,
                "auc": float(auc),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "confusion_matrix": cm.tolist(),
            },
            f,
            indent=2,
        )

    if single_class:
        print("Skipping ROC curve (single-class test set).")
    else:
        _save_roc_curve(test_labels, test_preds, auc)


def _save_roc_curve(labels, preds, auc):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(labels, preds)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("NeuroFlow Seizure Prediction ROC")
    plt.legend()
    plt.savefig(config.RESULTS_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    print("Saved", config.RESULTS_DIR / "roc_curve.png")


if __name__ == "__main__":
    main()
