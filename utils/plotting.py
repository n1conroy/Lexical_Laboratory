# utils/plotting.py

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def save_classification_report(y_true, y_pred, output_dir, labels=None):
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "classification_report.json")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"Classification report saved to {report_path}")
    return report


def save_confusion_matrix(y_true, y_pred, output_dir, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path)
    plt.close()

    print(f"Confusion matrix saved to {path}")
