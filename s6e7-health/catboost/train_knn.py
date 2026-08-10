"""Phase-2 approximate kNN member.

Uses a class-balanced reference subsample per fold and FAISS HNSW search. This is an
instance-based, local-neighborhood probe, not another tree learner.

Writes:
  catboost/oof__knn.csv
  catboost/test_proba__knn.csv
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / "catboost" / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT / "shared"))

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")

import faiss
import numpy as np
import pandas as pd

from metric import CLASSES, CLASS_TO_IDX, balanced_accuracy, labels_from_proba
from train_generative import make_features, make_preprocessor


N_FOLDS = 5
SEED = 42
PCOLS = ["p_at-risk", "p_fit", "p_unhealthy"]


def balanced_reference_indices(y, rng, max_per_class):
    keep = []
    for cls in CLASSES:
        idx = np.flatnonzero(y == cls)
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return keep


def proba_from_neighbors(distances, neighbor_labels, n_classes=3):
    distances = np.maximum(distances, 0.0)
    weights = 1.0 / (np.sqrt(distances) + 1e-3)
    out = np.zeros((neighbor_labels.shape[0], n_classes), dtype=np.float32)
    for c in range(n_classes):
        out[:, c] = (weights * (neighbor_labels == c)).sum(axis=1)
    out += 1e-6
    out /= out.sum(axis=1, keepdims=True)
    return out


def search_proba(index, queries, ref_y_idx, k, chunk_size):
    out = np.zeros((len(queries), len(CLASSES)), dtype=np.float32)
    for start in range(0, len(queries), chunk_size):
        end = min(start + chunk_size, len(queries))
        distances, neighbors = index.search(queries[start:end], k)
        out[start:end] = proba_from_neighbors(distances, ref_y_idx[neighbors])
    return out


def write_outputs(train_ids, test_ids, oof, test_proba):
    pd.DataFrame(
        {
            "id": train_ids,
            "p_at-risk": oof[:, 0],
            "p_fit": oof[:, 1],
            "p_unhealthy": oof[:, 2],
        }
    ).to_csv(ROOT / "catboost" / "oof__knn.csv", index=False)
    pd.DataFrame(
        {
            "id": test_ids,
            "p_at-risk": test_proba[:, 0],
            "p_fit": test_proba[:, 1],
            "p_unhealthy": test_proba[:, 2],
        }
    ).to_csv(ROOT / "catboost" / "test_proba__knn.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-class", type=int, default=30000)
    parser.add_argument("--neighbors", type=int, default=101)
    parser.add_argument("--chunk-size", type=int, default=25000)
    parser.add_argument("--hnsw-m", type=int, default=24)
    parser.add_argument("--ef-search", type=int, default=96)
    args = parser.parse_args()

    faiss.omp_set_num_threads(3)
    rng = np.random.default_rng(SEED)

    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    folds = pd.read_csv(ROOT / "shared" / "folds.csv").set_index("id")["fold"]
    train["fold"] = train["id"].map(folds).astype(int)
    assert train["fold"].notna().all() and set(train["fold"]) == set(range(N_FOLDS))

    y = train["health_condition"].to_numpy()
    y_idx_all = np.array([CLASS_TO_IDX[c] for c in y], dtype=np.int64)
    x_train = make_features(train)
    x_test = make_features(test)

    oof = np.zeros((len(train), len(CLASSES)), dtype=np.float32)
    test_proba = np.zeros((len(test), len(CLASSES)), dtype=np.float32)

    for fold in range(N_FOLDS):
        valid = train["fold"].to_numpy() == fold
        train_pos = np.flatnonzero(~valid)
        local_keep = balanced_reference_indices(y[~valid], rng, args.max_per_class)
        ref_pos = train_pos[local_keep]
        print(
            f"fold {fold}: reference={len(ref_pos)} valid={int(valid.sum())} "
            f"k={args.neighbors}",
            flush=True,
        )

        prep = make_preprocessor(x_train)
        x_ref = prep.fit_transform(x_train.iloc[ref_pos]).astype(np.float32, copy=False)
        x_va = prep.transform(x_train.loc[valid]).astype(np.float32, copy=False)
        x_te = prep.transform(x_test).astype(np.float32, copy=False)

        index = faiss.IndexHNSWFlat(x_ref.shape[1], args.hnsw_m, faiss.METRIC_L2)
        index.hnsw.efConstruction = 80
        index.hnsw.efSearch = args.ef_search
        index.add(np.ascontiguousarray(x_ref))

        ref_y_idx = y_idx_all[ref_pos]
        oof[valid] = search_proba(index, np.ascontiguousarray(x_va), ref_y_idx, args.neighbors, args.chunk_size)
        test_proba += search_proba(
            index,
            np.ascontiguousarray(x_te),
            ref_y_idx,
            args.neighbors,
            args.chunk_size,
        ) / N_FOLDS
        score = balanced_accuracy(y[valid], labels_from_proba(oof[valid]))
        print(f"fold {fold}: knn argmax balanced-accuracy {score:.6f}", flush=True)

    score = balanced_accuracy(y, labels_from_proba(oof))
    write_outputs(train["id"], test["id"], oof, test_proba)
    print(f"knn: OOF argmax balanced-accuracy {score:.6f}")
    print("wrote catboost/oof__knn.csv and catboost/test_proba__knn.csv")


if __name__ == "__main__":
    main()
