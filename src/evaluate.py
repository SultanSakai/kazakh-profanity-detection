"""Step 3: evaluate the saved model ONCE on the held-out test set.

Run:  python -m src.evaluate

Writes to results/:
  test_metrics.json      precision / recall / F1 for the profanity class, confusion matrix
  confusion_matrix.png   the same matrix as a picture
  test_predictions.csv   every test comment with its probability and prediction
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

from src.model import load_model

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def main():
    pipe, meta = load_model(ROOT / CFG["model_dir"])
    thr = meta["threshold"]
    test = pd.read_csv(ROOT / CFG["processed_dir"] / "test.csv")

    prob = pipe.predict_proba(test.text)[:, 1]
    pred = (prob >= thr).astype(int)
    y = test.label.values
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "n_test": int(len(test)),
        "n_test_label_0": int((y == 0).sum()),
        "n_test_label_1": int((y == 1).sum()),
        "threshold": thr,
        "profanity_class": {
            "precision": round(precision_score(y, pred), 4),
            "recall": round(recall_score(y, pred), 4),
            "f1": round(f1_score(y, pred), 4),
        },
        "accuracy": round(accuracy_score(y, pred), 4),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "classification_report": classification_report(
            y, pred, target_names=["0 clean", "1 profanity"], output_dict=True),
    }
    out = ROOT / CFG["results_dir"]
    (out / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["pred 0\nclean", "pred 1\nprofanity"])
    ax.set_yticks([0, 1], ["true 0\nclean", "true 1\nprofanity"])
    ax.set_title(f"Test confusion matrix (n={len(test)}, thr={thr})")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)

    test = test.assign(prob_profanity=prob.round(4), pred=pred,
                       error=["" if a == b else ("FP" if b == 1 else "FN") for a, b in zip(y, pred)])
    test.to_csv(out / "test_predictions.csv", index=False)

    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print(classification_report(y, pred, target_names=["0 clean", "1 profanity"], digits=3))


if __name__ == "__main__":
    main()
