"""Stage 3: turn pair probabilities into per-entity match lists.

1. One-to-one: in the ground truth every S2/S3 record belongs to at most one
   S1 entity, so each candidate is kept only for the S1 that scores it highest.
2. Per-entity selection, one of:
   - "threshold":  keep pairs with p >= t
   - "expected_f": pick the top-k (by p) maximizing expected F0.5, using
       F_beta = (1+b^2) TP / (b^2 |truth| + |pred|)
       E[F | top-k] ~= 1.25 * sum_{i<=k} p_i / (0.25 * sum_all p_i + k)
     and predict nothing when P(no match) = prod(1 - p_i) beats every k.
     Guarded by a probability floor `min_p`.
3. `tune` grid-searches the rule/parameters for best macro F0.5 on the
   validation split (singletons included, entities with no candidates count).
"""
import itertools

import polars as pl

from io_utils import log


def one_to_one(scored: pl.DataFrame, amb: float = 0.0) -> pl.DataFrame:
    """Keep each candidate only for its best-scoring S1.

    `amb` > 0: abstain entirely on a candidate whose two best S1 scores are
    within `amb` of each other -- e.g. two S1 entities with the same name and
    a candidate with an empty address. A coin-flip merge costs precision.
    """
    s = scored.with_columns(
        pl.col("p").rank("ordinal", descending=True).over("cand_idx").alias("_r"),
    )
    if amb > 0:
        second = pl.col("p").filter(pl.col("_r") == 2).first().over("cand_idx").fill_null(0.0)
        s = s.filter(pl.col("p") - second >= amb)
    return s.filter(pl.col("_r") == 1).drop("_r")


def select_threshold(scored: pl.DataFrame, t: float) -> pl.DataFrame:
    return scored.filter(pl.col("p") >= t)


def select_expected_f(scored: pl.DataFrame, min_p: float = 0.05, max_k: int = 15) -> pl.DataFrame:
    df = scored.filter(pl.col("p") >= min_p * 0.5).sort(["s1_idx", "p"], descending=[False, True])
    df = df.with_columns(
        pl.col("p").cum_sum().over("s1_idx").alias("_cum_p"),
        pl.col("p").sum().over("s1_idx").alias("_tot_p"),
        (1 - pl.col("p")).log().sum().over("s1_idx").exp().alias("_p_none"),
        pl.int_range(1, pl.len() + 1).over("s1_idx").alias("_k"),
    )
    df = df.with_columns((1.25 * pl.col("_cum_p") / (0.25 * pl.col("_tot_p") + pl.col("_k"))).alias("_ef"))
    df = df.with_columns(
        pl.col("_ef").max().over("s1_idx").alias("_best_ef"),
    )
    best_k = (
        df.filter(pl.col("_ef") == pl.col("_best_ef")).group_by("s1_idx").agg(pl.col("_k").min().alias("_best_k"))
    )
    df = df.join(best_k, on="s1_idx")
    keep = (pl.col("_k") <= pl.col("_best_k")) & (pl.col("_best_ef") > pl.col("_p_none")) \
        & (pl.col("p") >= min_p) & (pl.col("_k") <= max_k)
    return df.filter(keep).select(scored.columns)


def apply_rule(scored: pl.DataFrame, params: dict) -> pl.DataFrame:
    s = one_to_one(scored, params.get("amb", 0.0))
    if params["rule"] == "threshold":
        return select_threshold(s, params["t"])
    return select_expected_f(s, params["min_p"], params.get("max_k", 15))


def macro_f05(pred: pl.DataFrame, s1_meta: pl.DataFrame) -> float:
    """pred: s1_idx, label (1 if the predicted pair is a true match). s1_meta: s1_idx, n_true."""
    agg = pred.group_by("s1_idx").agg(pl.len().alias("n_pred"), pl.col("label").sum().alias("tp"))
    per = s1_meta.join(agg, on="s1_idx", how="left").with_columns(
        pl.col("n_pred").fill_null(0), pl.col("tp").fill_null(0)
    )
    f = (
        pl.when(pl.col("n_true") == 0).then((pl.col("n_pred") == 0).cast(pl.Float64))
        .when(pl.col("n_pred") == 0).then(0.0)
        .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("n_true") + pl.col("n_pred")))
    )
    return per.select(f.mean()).item()


def breakdown(pred: pl.DataFrame, s1_meta: pl.DataFrame) -> dict:
    """Precision/recall/F by singleton vs non-singleton, for error analysis."""
    agg = pred.group_by("s1_idx").agg(pl.len().alias("n_pred"), pl.col("label").sum().alias("tp"))
    per = s1_meta.join(agg, on="s1_idx", how="left").with_columns(
        pl.col("n_pred").fill_null(0), pl.col("tp").fill_null(0)
    )
    sing = per.filter(pl.col("n_true") == 0)
    non = per.filter(pl.col("n_true") > 0)
    return {
        "singleton_acc": round(float((sing["n_pred"] == 0).mean()), 4) if sing.height else None,
        "nonsingleton_empty_pred": round(float((non["n_pred"] == 0).mean()), 4),
        "pair_precision": round(float(per["tp"].sum() / max(1, per["n_pred"].sum())), 4),
        "pair_recall": round(float(per["tp"].sum() / max(1, per["n_true"].sum())), 4),
    }


def tune(scored_val: pl.DataFrame, s1_meta_val: pl.DataFrame) -> tuple:
    """Grid search over decision rules; returns (best_params, best_f, all_results)."""
    grid = []
    ambs = [0.0, 0.05, 0.15, 0.3]
    for t, a in itertools.product([0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], ambs):
        grid.append({"rule": "threshold", "t": t, "amb": a})
    for mp, a in itertools.product([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], ambs):
        grid.append({"rule": "expected_f", "min_p": mp, "amb": a})
    results = []
    for params in grid:
        pred = apply_rule(scored_val, params)
        f = macro_f05(pred, s1_meta_val)
        results.append((f, params))
    results.sort(key=lambda x: -x[0])
    best_f, best = results[0]
    log(f"decision tuning: best {best} -> val macro F0.5 = {best_f:.4f}")
    for f, p in results[:5]:
        log(f"    {f:.4f}  {p}")
    return best, best_f, results
