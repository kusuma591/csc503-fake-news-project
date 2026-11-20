import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

RESULTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "evaluation", "results")
)


def evaluate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }

def delta(shift_metrics, baseline_metrics):
    keys = ["accuracy", "precision", "recall", "f1"]
    return {k: round(shift_metrics[k] - baseline_metrics[k], 4) for k in keys}

def build_results_table(results, deltas):
    rows = []
    conditions = ["baseline", "temporal", "topic", "both"]

    for clf in results:
        for cond in conditions:
            m = results[clf][cond]
            row = {
                "Classifier": clf,
                "Condition": cond,
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1": m["f1"],
                "TP": m["TP"],
                "TN": m["TN"],
                "FP": m["FP"],
                "FN": m["FN"],
            }
            if cond != "baseline" and clf in deltas and cond in deltas[clf]:
                d = deltas[clf][cond]
                row["Delta_Accuracy"] = d["accuracy"]
                row["Delta_Precision"] = d["precision"]
                row["Delta_Recall"] = d["recall"]
                row["Delta_F1"] = d["f1"]
            else:
                row["Delta_Accuracy"] = "-"
                row["Delta_Precision"] = "-"
                row["Delta_Recall"] = "-"
                row["Delta_F1"] = "-"
            rows.append(row)

    return pd.DataFrame(rows)


def save_results(df, output_cm=False, filename="results.csv"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")
    
    if not output_cm: return

    for _, row in df.iterrows():
        if row["TP"] + row["TN"] + row["FP"] + row["FN"] == 0:
            continue

        cm = np.array([
            [row["TN"], row["FP"]],
            [row["FN"], row["TP"]],
        ])

        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Not Fake", "Fake"],
        )
        disp.plot(ax=ax, cmap="Blues", colorbar=True)
        ax.set_title(f"{row['Classifier']} - {row['Condition']}")

        clf_slug = row["Classifier"].lower().replace(" ", "_")
        fig_filename = f"cm_{clf_slug}_{row['Condition']}.png"
        save_path = os.path.join(RESULTS_DIR, "confusion_matrices", fig_filename)

        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Confusion matrix saved to {save_path}")
