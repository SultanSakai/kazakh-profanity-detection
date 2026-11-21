"""Step 2: choose settings on the validation set and train the final model.

Run:  python -m src.train

* Every candidate is trained on TRAIN only and scored on VAL.
* The best candidate = highest validation F1 for class 1 (profanity) at the
  default 0.5 threshold; ties -> higher average precision -> smaller C.
* The decision threshold is then tuned on VAL (maximize F1, ties -> closest to 0.5).
* The test set is NOT touched here.
"""

import itertools
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from src.model import build_pipeline, save_model

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def scores(y, p, thr):
    pred = (p >= thr).astype(int)
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }


def main():
    seed = CFG["seed"]
    np.random.seed(seed)
    data = ROOT / CFG["processed_dir"]
    train = pd.read_csv(data / "train.csv")
    val = pd.read_csv(data / "val.csv")
    print(f"train={len(train)}  val={len(val)}")

    space = CFG["search_space"]
    rows = []
    for norm, ngr, C in itertools.product(space["normalize"], space["ngram_range"], space["C"]):
        pipe = build_pipeline(norm, ngr, C, seed, CFG["fixed_vectorizer"])
        pipe.fit(train.text, train.label)
        p = pipe.predict_proba(val.text)[:, 1]
        r = {"normalize": norm, "ngram_range": str(tuple(ngr)), "C": C,
             **scores(val.label, p, 0.5),
             "average_precision": average_precision_score(val.label, p),
             "n_features": len(pipe.named_steps["tfidf"].vocabulary_)}
        rows.append(r)
        print(f"norm={norm!s:5} ngram={r['ngram_range']} C={C:<4} "
              f"F1={r['f1']:.3f} P={r['precision']:.3f} R={r['recall']:.3f} AP={r['average_precision']:.3f}")

    res = pd.DataFrame(rows).sort_values(
        ["f1", "average_precision", "C"], ascending=[False, False, True]).reset_index(drop=True)
    out = ROOT / CFG["results_dir"]
    out.mkdir(exist_ok=True)
    res.round(4).to_csv(out / "val_search.csv", index=False)
    best = res.iloc[0]
    print("\nBest settings on validation:\n", best.to_string())

    # Effect of normalization alone (best run with / without it), for the README.
    ablation = res.groupby("normalize").first()[["ngram_range", "C", "f1", "average_precision"]]
    print("\nBest run with and without normalization:\n", ablation)

    # Refit the chosen configuration on TRAIN and tune the threshold on VAL.
    cfg = {"normalize": bool(best.normalize),
           "ngram_range": [int(v) for v in best.ngram_range.strip("()").split(",")],
           "C": float(best.C)}
    pipe = build_pipeline(cfg["normalize"], cfg["ngram_range"], cfg["C"], seed, CFG["fixed_vectorizer"])
    pipe.fit(train.text, train.label)
    p_val = pipe.predict_proba(val.text)[:, 1]

    g = CFG["threshold_grid"]
    thr_rows = []
    for thr in np.round(np.arange(g["start"], g["stop"] + 1e-9, g["step"]), 2):
        thr_rows.append({"threshold": float(thr), **scores(val.label, p_val, thr)})
    thr_df = pd.DataFrame(thr_rows)
    thr_df.round(4).to_csv(out / "val_threshold_curve.csv", index=False)
    top = thr_df[thr_df.f1 == thr_df.f1.max()].copy()
    top["dist"] = (top.threshold - 0.5).abs()
    threshold = float(top.sort_values("dist").iloc[0].threshold)
    val_at_thr = scores(val.label, p_val, threshold)
    print(f"\nChosen threshold = {threshold}  (val F1={val_at_thr['f1']:.3f}, "
          f"P={val_at_thr['precision']:.3f}, R={val_at_thr['recall']:.3f})")

    meta = {
        "model": "char n-gram TF-IDF + logistic regression",
        "settings": cfg,
        "vectorizer": CFG["fixed_vectorizer"],
        "threshold": threshold,
        "seed": seed,
        "n_features": len(pipe.named_steps["tfidf"].vocabulary_),
        "train_rows": len(train),
        "val_metrics_at_threshold": {k: round(v, 4) for k, v in val_at_thr.items()},
        "val_average_precision": round(float(average_precision_score(val.label, p_val)), 4),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    save_model(pipe, meta, ROOT / CFG["model_dir"])
    print("\nSaved model to", ROOT / CFG["model_dir"])


if __name__ == "__main__":
    main()
