from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# --------------------------------------------------
# Paths
# --------------------------------------------------

gt_dir = Path("/home/xgupke/Documents/projects/DeepPI-EM/data_Mut_Ctrl/test/target")
pred_dir = Path("/home/xgupke/Documents/projects/DeepPI-EM/runs/mito_mut_ctrl/test_results_7030")

individual_csv = pred_dir / "image_mask_metrics.csv"
class_csv = pred_dir / "class_cumulative_metrics.csv"


# --------------------------------------------------
# Get corresponding files
# --------------------------------------------------

gt_files = sorted(gt_dir.glob("*.png"))
pred_files = sorted(pred_dir.glob("*.png"))

assert len(gt_files) == len(pred_files), (
    f"Different number of masks: "
    f"{len(gt_files)} ground truth vs {len(pred_files)} predicted"
)


# --------------------------------------------------
# Metric calculation
# --------------------------------------------------

def calculate_metrics(gt, pred):

    # Convert masks to binary
    gt = gt > 0
    pred = pred > 0

    # Confusion matrix
    tp = np.sum(gt & pred)
    tn = np.sum(~gt & ~pred)
    fp = np.sum(~gt & pred)
    fn = np.sum(gt & ~pred)

    # Metrics
    iou = (
        tp / (tp + fp + fn)
        if (tp + fp + fn) != 0
        else 1.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) != 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) != 0
        else 0.0
    )

    f1 = (
        2 * tp / (2 * tp + fp + fn)
        if (2 * tp + fp + fn) != 0
        else 0.0
    )

    # True positive rate (sensitivity)
    tpr = recall

    # False positive rate
    fpr = (
        fp / (fp + tn)
        if (fp + tn) != 0
        else 0.0
    )

    # True negative rate (specificity)
    tnr = (
        tn / (tn + fp)
        if (tn + fp) != 0
        else 0.0
    )

    # False negative rate
    fnr = (
        fn / (fn + tp)
        if (fn + tp) != 0
        else 0.0
    )

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tpr": tpr,
        "tnr": tnr,
        "fpr": fpr,
        "fnr": fnr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# --------------------------------------------------
# Process all images
# --------------------------------------------------

results = []

for gt_path, pred_path in zip(gt_files, pred_files):

    gt = np.array(Image.open(gt_path))
    pred = np.array(Image.open(pred_path))

    if gt.shape != pred.shape:
        raise ValueError(
            f"Shape mismatch:\n"
            f"{gt_path.name}: {gt.shape}\n"
            f"{pred_path.name}: {pred.shape}"
        )

    metrics = calculate_metrics(gt, pred)

    results.append({
        "ground_truth_filename": gt_path.name,
        "prediction_filename": pred_path.name,
        "class": "Mut" if "mut" in gt_path.name.lower() else "Control",
        **metrics,
    })


# --------------------------------------------------
# Save individual image metrics
# --------------------------------------------------

df = pd.DataFrame(results)

individual_columns = [
    "ground_truth_filename",
    "prediction_filename",
    "class",
    "iou",
    "precision",
    "recall",
    "f1_score",
    "tpr",
    "tnr",
    "fpr",
    "fnr",
    "tp",
    "tn",
    "fp",
    "fn",
]

df = df[individual_columns]

df.to_csv(individual_csv, index=False)

print("\nIndividual image metrics:")
print(df)

print(f"\nSaved individual metrics to:")
print(individual_csv)


# --------------------------------------------------
# Create class-wise cumulative metrics
# --------------------------------------------------

class_results = []

for class_name, group in df.groupby("class"):

    # Sum confusion matrices
    TP = group["tp"].sum()
    TN = group["tn"].sum()
    FP = group["fp"].sum()
    FN = group["fn"].sum()

    iou = (
        TP / (TP + FP + FN)
        if (TP + FP + FN) != 0
        else 1.0
    )

    precision = (
        TP / (TP + FP)
        if (TP + FP) != 0
        else 0.0
    )

    recall = (
        TP / (TP + FN)
        if (TP + FN) != 0
        else 0.0
    )

    f1 = (
        2 * TP / (2 * TP + FP + FN)
        if (2 * TP + FP + FN) != 0
        else 0.0
    )

    tpr = recall

    tnr = (
        TN / (TN + FP)
        if (TN + FP) != 0
        else 0.0
    )

    fpr = (
        FP / (FP + TN)
        if (FP + TN) != 0
        else 0.0
    )

    fnr = (
        FN / (FN + TP)
        if (FN + TP) != 0
        else 0.0
    )

    class_results.append({
        "filename": class_name,
        "class": class_name,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tpr": tpr,
        "tnr": tnr,
        "fpr": fpr,
        "fnr": fnr,
        "tp": TP,
        "tn": TN,
        "fp": FP,
        "fn": FN,
    })


class_df = pd.DataFrame(class_results)

class_df = class_df.drop("filename", axis=1)

class_df.to_csv(class_csv, index=False)

print("\nClass-wise cumulative metrics:")
print(class_df)

print(f"\nSaved class-wise metrics to:")
print(class_csv)