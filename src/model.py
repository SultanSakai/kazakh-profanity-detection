"""Model definition, saving/loading and explanation helpers."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.text_normalization import basic_clean_batch, normalize_batch

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = "profanity_pipeline.joblib"
META_FILE = "model_meta.json"


def build_pipeline(normalize, ngram_range, C, seed, vec_cfg):
    """normalize -> character n-gram TF-IDF -> logistic regression.

    No class weighting: the training set is balanced (about 50% / 50%).
    """
    prep = normalize_batch if normalize else basic_clean_batch
    return Pipeline([
        ("prep", FunctionTransformer(prep, validate=False)),
        ("tfidf", TfidfVectorizer(
            analyzer=vec_cfg["analyzer"],
            ngram_range=tuple(ngram_range),
            min_df=vec_cfg["min_df"],
            sublinear_tf=vec_cfg["sublinear_tf"],
            lowercase=False,          # already lower-cased in "prep"
        )),
        ("clf", LogisticRegression(
            C=C, max_iter=5000,
            solver="liblinear", random_state=seed,
        )),
    ])


def save_model(pipe, meta, model_dir=None):
    model_dir = Path(model_dir or ROOT / "models")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_dir / MODEL_FILE, compress=3)
    (model_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_model(model_dir=None):
    model_dir = Path(model_dir or ROOT / "models")
    pipe = joblib.load(model_dir / MODEL_FILE)
    meta = json.loads((model_dir / META_FILE).read_text(encoding="utf-8"))
    return pipe, meta


def explain(pipe, text, top_k=8):
    """Return the n-grams that pushed this comment's score up or down the most.

    For logistic regression the score is  z = b + sum_i w_i * x_i  and the
    probability is sigmoid(z). Each n-gram's contribution is simply w_i * x_i.
    """
    prepped = pipe.named_steps["prep"].transform([text])[0]
    x = pipe.named_steps["tfidf"].transform([prepped])
    clf = pipe.named_steps["clf"]
    vocab = pipe.named_steps["tfidf"].get_feature_names_out()
    idx = x.indices
    contrib = x.data * clf.coef_[0, idx]
    order = np.argsort(contrib)
    top_pos = [(vocab[idx[i]], float(contrib[i])) for i in order[::-1][:top_k] if contrib[i] > 0]
    top_neg = [(vocab[idx[i]], float(contrib[i])) for i in order[:top_k] if contrib[i] < 0]
    z = float(clf.intercept_[0] + contrib.sum())
    return {
        "normalized_text": prepped,
        "n_active_ngrams": int(x.nnz),
        "intercept": float(clf.intercept_[0]),
        "score_z": z,
        "probability": float(1 / (1 + np.exp(-z))),
        "pushes_to_profane": top_pos,
        "pushes_to_clean": top_neg,
    }
