"""Evaluate the trained CfC model on the held-out test patients.

Reports the clinical KPIs -- per-seizure sensitivity and event-wise false alarms
per hour -- at a decision threshold tuned on the validation set to the FA/h
target, plus per-window AUROC/AUPRC. Writes results/evaluation.json and
results/roc_curve.png.

Run:  python evaluate.py [path/to/checkpoint.ckpt]
"""

import json
import sys
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset

import config
from model import CfCSeizurePredictor


def find_checkpoint(explicit=None):
    """Best checkpoint: explicit arg, else newest cfc100-*.ckpt (ModelCheckpoint
    keeps only the single best via save_top_k=1), else last.ckpt."""
    if explicit:
        return explicit
    ckpts = sorted(config.MODEL_DIR.glob("cfc100-*.ckpt"), key=lambda p: p.stat().st_mtime)
    if ckpts:
        return str(ckpts[-1])
    last = config.MODEL_DIR / "last.ckpt"
    if last.exists():
        return str(last)
    raise SystemExit(f"No checkpoint found in {config.MODEL_DIR}. Train first.")


def _subset_tensor(segments_mm, idx):
    """Materialise a memmap subset as a float tensor (keeps RAM bounded)."""
    return torch.from_numpy(np.ascontiguousarray(segments_mm[idx])).float()


def predict_in_batches(model, segments, device):
    loader = DataLoader(TensorDataset(segments), batch_size=config.BATCH_SIZE)
    preds = []
    for (batch,) in loader:
        preds.append(model.predict_proba(batch.to(device)).cpu())
    return torch.cat(preds).numpy()


def _fa_per_hour(labels, preds, thr):
    """False alarms per hour over interictal epochs (epoch-rate proxy)."""
    interictal = labels == 0
    n = int(interictal.sum())
    if n == 0:
        return 0.0
    fp = int(((preds >= thr) & interictal).sum())
    hours = n * config.EPOCH_SECONDS / 3600.0
    return fp / hours if hours else 0.0


def _tune_threshold(labels, preds, target_fa):
    """Lowest threshold (max sensitivity) whose FA/h <= target on the val set."""
    best = 0.99
    for thr in np.linspace(0.99, 0.01, 99):
        if _fa_per_hour(labels, preds, float(thr)) <= target_fa:
            best = float(thr)
        else:
            break  # FA/h only rises as the threshold drops
    return best


def _per_patient_metrics(labels, preds, patient_ids):
    """Subject-wise AUROC/AUPRC, the clinically meaningful frame.

    Pooling patients with different preictal base rates inflates the aggregate
    AUROC (it partly measures *which patient* this is, not *is this preictal*). A
    deployed predictor runs within one patient, so we report each patient's own
    AUROC and the macro-average. Single-class patients are skipped (AUROC is
    undefined). Returns (per_patient: {pid: {auroc, auprc, n, preictal}}, mean_auroc).
    """
    per = {}
    by_pid = defaultdict(list)
    for pid, p, y in zip(patient_ids, preds, labels, strict=True):
        by_pid[pid].append((p, int(y)))
    for pid, rows in sorted(by_pid.items()):
        yy = np.array([y for _, y in rows])
        pp = np.array([p for p, _ in rows])
        if len(np.unique(yy)) < 2:
            continue
        per[pid] = {
            "auroc": float(roc_auc_score(yy, pp)),
            "auprc": float(average_precision_score(yy, pp)),
            "n": int(len(yy)),
            "preictal": int(yy.sum()),
        }
    mean_auroc = float(np.mean([v["auroc"] for v in per.values()])) if per else float("nan")
    return per, mean_auroc


def _per_seizure_sensitivity(labels, preds, groups, thr):
    """Fraction of seizures with >=1 preictal epoch above thr -> (sens, hit, total)."""
    by_seizure = defaultdict(list)
    for g, p, y in zip(groups, preds, labels, strict=True):
        if y == 1 and g:  # preictal epoch carrying a seizure group id
            by_seizure[g].append(p)
    if not by_seizure:
        return float("nan"), 0, 0
    hit = sum(1 for probs in by_seizure.values() if max(probs) >= thr)
    return hit / len(by_seizure), hit, len(by_seizure)


def main():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = find_checkpoint(sys.argv[1] if len(sys.argv) > 1 else None)
    print("Loading", ckpt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CfCSeizurePredictor.load_from_checkpoint(ckpt, map_location=device).to(device)
    model.eval()

    segments = np.load(config.OUTPUT_DIR / "segments.npy", mmap_mode="r")
    labels = np.load(config.OUTPUT_DIR / "labels.npy")
    with open(config.OUTPUT_DIR / "seizure_groups.json") as f:
        groups = np.array(json.load(f), dtype=object)
    with open(config.OUTPUT_DIR / "split.json") as f:
        split = json.load(f)
    with open(config.OUTPUT_DIR / "patient_ids.json") as f:
        patient_ids = np.array(json.load(f), dtype=object)
    val_idx, test_idx = split["val_idx"], split["test_idx"]

    val_labels = labels[val_idx]
    val_preds = predict_in_batches(model, _subset_tensor(segments, val_idx), device)
    test_labels = labels[test_idx]
    test_preds = predict_in_batches(model, _subset_tensor(segments, test_idx), device)
    test_groups = groups[test_idx]

    # Decision threshold: explicit override, else tuned on val to the FA/h target.
    if config.EVAL_THRESHOLD is not None:
        thr = config.EVAL_THRESHOLD
        print(f"Decision threshold: {thr:.3f} (fixed via NEUROFLOW_THRESHOLD)")
    else:
        thr = _tune_threshold(val_labels, val_preds, config.EVAL_TARGET_FA_PER_H)
        print(
            f"Decision threshold: {thr:.3f} (tuned on val for <= {config.EVAL_TARGET_FA_PER_H} FA/h)"
        )

    both_classes = len(np.unique(test_labels)) > 1
    auroc = float(roc_auc_score(test_labels, test_preds)) if both_classes else float("nan")
    auprc = (
        float(average_precision_score(test_labels, test_preds)) if both_classes else float("nan")
    )

    cm = confusion_matrix(test_labels, (test_preds >= thr).astype(int), labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    win_sens = tp / (tp + fn) if (tp + fn) else 0.0
    win_spec = tn / (tn + fp) if (tn + fp) else 0.0
    fa_h = _fa_per_hour(test_labels, test_preds, thr)
    seiz_sens, hit, total = _per_seizure_sensitivity(test_labels, test_preds, test_groups, thr)
    per_patient, mean_patient_auroc = _per_patient_metrics(
        test_labels, test_preds, patient_ids[test_idx]
    )

    print(f"\n== Test metrics @ threshold {thr:.3f} ==")
    print(f"Per-seizure sensitivity: {seiz_sens:.3f}  ({hit}/{total} seizures)")
    print(f"False alarms / hour:     {fa_h:.3f}  (target < {config.EVAL_TARGET_FA_PER_H})")
    print(f"Pooled AUROC: {auroc:.3f} | AUPRC: {auprc:.3f}")
    print(f"Per-window sensitivity: {win_sens:.3f} | specificity: {win_spec:.3f}")
    print(f"Confusion [[tn fp][fn tp]]:\n{cm}")
    # Subject-wise AUROC is the honest metric: the pooled value above is inflated
    # by differing per-patient preictal base rates.
    print(f"\nMean per-patient AUROC: {mean_patient_auroc:.3f}  (clinically meaningful)")
    for pid, m in per_patient.items():
        print(
            f"  {pid}: AUROC {m['auroc']:.3f} | AUPRC {m['auprc']:.3f} "
            f"({m['preictal']}/{m['n']} preictal)"
        )

    with open(config.RESULTS_DIR / "evaluation.json", "w") as f:
        json.dump(
            {
                "checkpoint": ckpt,
                "threshold": thr,
                "target_fa_per_h": config.EVAL_TARGET_FA_PER_H,
                "per_seizure_sensitivity": seiz_sens,
                "seizures_predicted": hit,
                "seizures_total": total,
                "false_alarms_per_hour": fa_h,
                "auroc": auroc,
                "auprc": auprc,
                "mean_per_patient_auroc": mean_patient_auroc,
                "per_patient": per_patient,
                "window_sensitivity": win_sens,
                "window_specificity": win_spec,
                "confusion_matrix": cm.tolist(),
                "n_test_epochs": int(len(test_labels)),
                "n_test_preictal": int((test_labels == 1).sum()),
            },
            f,
            indent=2,
        )

    if both_classes:
        _save_roc_curve(test_labels, test_preds, auroc)
    else:
        print("Skipping ROC curve (single-class test set).")


def _save_roc_curve(labels, preds, auc):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(labels, preds)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUROC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("NeuroFlow Seizure Prediction ROC")
    plt.legend()
    plt.savefig(config.RESULTS_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    print("Saved", config.RESULTS_DIR / "roc_curve.png")


if __name__ == "__main__":
    main()
