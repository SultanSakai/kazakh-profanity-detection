"""Step 4: predict on new comments with the saved model (no retraining).

Examples
  python -m src.predict "Өте жақсы тауар, рахмет!"
  python -m src.predict "Қотақбас ойын" --explain
  python -m src.predict            # interactive mode: type comments, empty line to quit

In Python / a notebook:
  from src.predict import predict
  predict(["Сапасы нашар екен", "мал ойын"])
"""

import argparse
import sys
from pathlib import Path

from src.model import explain, load_model

ROOT = Path(__file__).resolve().parents[1]
_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model(ROOT / "models")
    return _MODEL


def predict(texts):
    """Return a list of dicts: text, label (0/1), probability of profanity."""
    if isinstance(texts, str):
        texts = [texts]
    pipe, meta = _get_model()
    probs = pipe.predict_proba(texts)[:, 1]
    return [{"text": t, "label": int(p >= meta["threshold"]), "prob_profanity": round(float(p), 4)}
            for t, p in zip(texts, probs)]


def _print(text, show_explanation):
    r = predict(text)[0]
    name = "1 = Kazakh profanity" if r["label"] else "0 = no profanity"
    print(f"\n{text}\n  -> {name}   (probability of profanity = {r['prob_profanity']:.3f}, "
          f"threshold = {_get_model()[1]['threshold']})")
    if show_explanation:
        e = explain(_get_model()[0], text)
        print(f"  normalized: {e['normalized_text']!r}")
        print(f"  score z = intercept {e['intercept']:+.3f} + sum of n-gram weights = {e['score_z']:+.3f}")
        print("  pushes towards profanity:", ", ".join(f"'{g}' {c:+.2f}" for g, c in e["pushes_to_profane"]) or "-")
        print("  pushes towards clean:    ", ", ".join(f"'{g}' {c:+.2f}" for g, c in e["pushes_to_clean"]) or "-")


def main():
    ap = argparse.ArgumentParser(description="Kazakh profanity detector")
    ap.add_argument("texts", nargs="*", help="comment(s) to classify")
    ap.add_argument("--explain", action="store_true", help="show which n-grams drove the decision")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Kazakh letters in a Windows console
    if args.texts:
        for t in args.texts:
            _print(t, args.explain)
        return
    print("Type a comment and press Enter (empty line to quit).")
    while True:
        try:
            t = input("> ").strip()
        except EOFError:
            break
        if not t:
            break
        _print(t, args.explain)


if __name__ == "__main__":
    main()
