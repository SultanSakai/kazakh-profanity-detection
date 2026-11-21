"""Step 1: clean the labeled data and create leak-free train/val/test splits.

Run:  python -m src.prepare_data

What it does
1. Loads the shared KazSAnDRA-based file (kazsandra_id, text, label).
2. Removes leftover personal information (e-mails, URLs, phone numbers, @handles).
3. Applies the documented label corrections from data/label_corrections.csv.
4. Removes exact duplicates (after normalization: case, punctuation, elongation).
5. Groups near-duplicates (Jaccard similarity of character 3-gram sets) so that
   a group never ends up in two different splits.
6. Splits 80/10/10 with StratifiedGroupKFold (stratified by label, grouped by
   near-duplicate group, fixed random seed).
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedGroupKFold

from src.text_normalization import dedup_key

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

_EMAIL = re.compile(r"\S+@\S+\.\w+")
_URL = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)
_PHONE = re.compile(r"(\+?7|8)[\s\-(]*7\d{2}[\s\-)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}")
_HANDLE = re.compile(r"(?<![\w@])@(?!user\b)\w+")
_UNDERSCORE_HANDLE = re.compile(r"\b_[A-Za-z0-9]+_\b")


def scrub_pii(text: str) -> str:
    text = _EMAIL.sub("<EMAIL>", text)
    text = _URL.sub("<URL>", text)
    text = _PHONE.sub("<PHONE>", text)
    text = _HANDLE.sub("@user", text)
    text = _UNDERSCORE_HANDLE.sub("@user", text)
    return text


def near_duplicate_groups(keys, threshold, n):
    """Union-find over pairs whose character n-gram sets overlap >= threshold."""
    vec = CountVectorizer(analyzer="char_wb", ngram_range=(n, n), binary=True)
    X = csr_matrix(vec.fit_transform(keys), dtype=np.float32)
    sizes = np.asarray(X.sum(axis=1)).ravel()
    inter = (X @ X.T).toarray()
    union = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(union > 0, inter / union, 0.0)
    np.fill_diagonal(jac, 0.0)

    parent = list(range(len(keys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    rows, cols = np.where(np.triu(jac) >= threshold)
    for i, j in zip(rows, cols):
        parent[find(i)] = find(j)
    roots = [find(i) for i in range(len(keys))]
    remap = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return np.array([remap[r] for r in roots]), list(zip(rows, cols, jac[rows, cols]))


def main():
    seed = CFG["seed"]
    df = pd.read_excel(ROOT / CFG["raw_data"])
    df = df.rename(columns={"split": "original_split"})
    n_raw = len(df)
    stats = {"raw_rows": n_raw, "raw_class_counts": df.label.value_counts().sort_index().to_dict()}

    # 2. personal information
    scrubbed = df.text.map(scrub_pii)
    stats["rows_with_pii_scrubbed"] = int((scrubbed != df.text).sum())
    df["text"] = scrubbed

    # 3. label corrections
    corr = pd.read_csv(ROOT / CFG["label_corrections"])
    for _, c in corr.iterrows():
        mask = df.kazsandra_id == c.kazsandra_id
        assert mask.sum() == 1 and df.loc[mask, "label"].item() == c.old_label, c.kazsandra_id
        df.loc[mask, "label"] = c.new_label
    stats["label_corrections"] = len(corr)

    # 4. exact duplicates (after normalization)
    df["dedup_key"] = df.text.map(dedup_key)
    conflicts = df.groupby("dedup_key").label.nunique()
    assert (conflicts <= 1).all(), "duplicates with conflicting labels"
    dup_mask = df.dedup_key.duplicated(keep="first")
    stats["exact_duplicates_removed"] = int(dup_mask.sum())
    df = df[~dup_mask].reset_index(drop=True)

    # 5. near-duplicate groups
    groups, pairs = near_duplicate_groups(
        df.dedup_key.tolist(),
        CFG["split"]["near_duplicate_jaccard"],
        CFG["split"]["near_duplicate_char_ngram"],
    )
    df["group_id"] = groups
    gsize = df.group_id.map(df.group_id.value_counts())
    stats["near_duplicate_pairs"] = len(pairs)
    stats["near_duplicate_groups_size_gt1"] = int(df.loc[gsize > 1, "group_id"].nunique())
    stats["rows_in_near_duplicate_groups"] = int((gsize > 1).sum())

    # 6. grouped + stratified 80/10/10 split
    sgkf = StratifiedGroupKFold(n_splits=CFG["split"]["n_folds"], shuffle=True, random_state=seed)
    folds = list(sgkf.split(df, df.label, df.group_id))
    split = np.array(["train"] * len(df), dtype=object)
    split[folds[0][1]] = "test"     # 1 fold of 10 -> ~10% test
    split[folds[1][1]] = "val"      # 1 fold of 10 -> ~10% validation, other 8 folds -> train
    df["split"] = split
    assert df.groupby("group_id").split.nunique().max() == 1, "group leaks across splits"

    df["source"] = df.kazsandra_id.str[:3]
    out_dir = ROOT / CFG["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["kazsandra_id", "source", "text", "label", "group_id"]
    for name in ["train", "val", "test"]:
        df.loc[df.split == name, cols].to_csv(out_dir / f"{name}.csv", index=False)

    # near-duplicate pairs for inspection
    pd.DataFrame(
        [(df.text[i], df.text[j], round(float(s), 3)) for i, j, s in pairs],
        columns=["text_a", "text_b", "jaccard"],
    ).sort_values("jaccard").to_csv(out_dir / "near_duplicate_pairs.csv", index=False)

    stats["final_rows"] = len(df)
    stats["final_class_counts"] = df.label.value_counts().sort_index().to_dict()
    stats["splits"] = {
        name: {"rows": int((df.split == name).sum()),
               "label_0": int(((df.split == name) & (df.label == 0)).sum()),
               "label_1": int(((df.split == name) & (df.label == 1)).sum())}
        for name in ["train", "val", "test"]
    }
    stats["source_counts"] = df.source.value_counts().to_dict()
    stats["seed"] = seed
    stats = json.loads(json.dumps(stats, default=int))
    (ROOT / CFG["results_dir"]).mkdir(exist_ok=True)
    (ROOT / CFG["results_dir"] / "data_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
