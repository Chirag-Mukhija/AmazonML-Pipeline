"""Stage 2b: train the pairwise match classifier (LightGBM, MIT license).

Trains on candidate pairs of the "fit" split, early-stops on the "val" split.
Optional negative down-sampling (`--neg-frac`) for very large runs; the
sampling rate is saved with the model so predicted probabilities can be
re-calibrated (needed by the expected-F0.5 decision rule).
"""
import argparse
import glob
import json
import os

import lightgbm as lgb
import numpy as np
import polars as pl

from features import feature_columns
from io_utils import log

DEFAULT_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "metric": ["binary_logloss", "auc"],
    "verbosity": -1,
    "num_threads": 0,
}


def _parts(work_dir: str, dataset: str) -> list:
    return sorted(glob.glob(os.path.join(work_dir, dataset, "features", "part-*.parquet")))


def load_matrix(work_dir: str, dataset: str, folds, neg_frac: float = 1.0, seed: int = 0, cols=None):
    """(X float32, y int8, cols) built part by part so only one copy is in memory."""
    xs, ys = [], []
    for p in _parts(work_dir, dataset):
        f = pl.read_parquet(p).filter(pl.col("fold").is_in(folds))
        if neg_frac < 1.0:
            keep = (pl.col("label") == 1) | ((pl.col("cand_idx").hash(seed) % 10_000) < int(neg_frac * 10_000))
            f = f.filter(keep)
        if cols is None:
            cols = feature_columns(f)
        xs.append(f.select(cols).to_numpy().astype(np.float32))
        ys.append(f["label"].to_numpy().astype(np.int8))
        del f
    return np.concatenate(xs), np.concatenate(ys), cols


def correct_probs(p: np.ndarray, neg_frac: float) -> np.ndarray:
    """Undo negative down-sampling: odds_true = odds_model * neg_frac."""
    if neg_frac >= 1.0:
        return p
    odds = p / np.clip(1 - p, 1e-9, None) * neg_frac
    return odds / (1 + odds)


def train(work_dir: str, model_dir: str, neg_frac: float = 1.0, rounds: int = 3000,
          early_stop: int = 100, params: dict = None):
    os.makedirs(model_dir, exist_ok=True)
    params = {**DEFAULT_PARAMS, **(params or {})}
    Xf, yf, cols = load_matrix(work_dir, "train", ["fit"], neg_frac=neg_frac)
    # early stopping on down-sampled val too (same distribution as train); the decision
    # rule is tuned later on the full, un-sampled val set with re-calibrated probabilities
    Xv, yv, _ = load_matrix(work_dir, "train", ["val"], neg_frac=neg_frac, cols=cols)
    log(f"train rows={len(yf)} (pos={int(yf.sum())}), val rows={len(yv)}, features={len(cols)}")

    dtrain = lgb.Dataset(Xf, label=yf, feature_name=cols, free_raw_data=True)
    dval = lgb.Dataset(Xv, label=yv, reference=dtrain, feature_name=cols)
    del Xf, Xv
    booster = lgb.train(
        params, dtrain, num_boost_round=rounds, valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(early_stop, verbose=False), lgb.log_evaluation(100)],
    )
    booster.save_model(os.path.join(model_dir, "model.txt"))
    imp = pl.DataFrame({
        "feature": cols,
        "gain": booster.feature_importance("gain"),
        "split": booster.feature_importance("split"),
    }).sort("gain", descending=True)
    imp.write_csv(os.path.join(model_dir, "feature_importance.csv"))
    meta = {"features": cols, "neg_frac": neg_frac, "best_iteration": booster.best_iteration,
            "params": params}
    with open(os.path.join(model_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"saved model (best_iteration={booster.best_iteration}); top features: "
        f"{imp.head(10)['feature'].to_list()}")
    return booster


def predict(work_dir: str, dataset: str, model_dir: str, folds=None) -> pl.DataFrame:
    """Score all featurized pairs -> s1_idx, cand_idx, p (+ label/fold if train)."""
    booster = lgb.Booster(model_file=os.path.join(model_dir, "model.txt"))
    with open(os.path.join(model_dir, "model_meta.json")) as f:
        meta = json.load(f)
    cols = meta["features"]
    out = []
    for path in _parts(work_dir, dataset):
        f = pl.read_parquet(path)
        if folds is not None and "fold" in f.columns:
            f = f.filter(pl.col("fold").is_in(folds))
        missing = [c for c in cols if c not in f.columns]
        if missing:
            raise ValueError(f"features missing at inference: {missing}")
        p = booster.predict(f.select(cols).to_numpy().astype(np.float32), num_iteration=booster.best_iteration)
        p = correct_probs(p, meta.get("neg_frac", 1.0))
        keep = ["s1_idx", "cand_idx"] + [c for c in ("label", "fold") if c in f.columns]
        out.append(f.select(keep).with_columns(pl.Series("p", p.astype(np.float32))))
    return pl.concat(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--neg-frac", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--early-stop", type=int, default=100)
    ap.add_argument("--learning-rate", type=float, default=DEFAULT_PARAMS["learning_rate"])
    args = ap.parse_args()
    train(args.work_dir, args.model_dir, args.neg_frac, args.rounds, args.early_stop,
          {"learning_rate": args.learning_rate})


if __name__ == "__main__":
    main()
