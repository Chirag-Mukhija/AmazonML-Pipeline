"""Stage 2a: pairwise + contextual features for every candidate pair.

Everything is vectorized: string similarities run through rapidfuzz's
multithreaded `cpdist` (element-wise over aligned arrays), set overlaps
through polars list ops. Features are deliberately country-agnostic (no
country one-hot) so the model transfers to France, which is absent in train.

Context features exploit the data's structure -- every S2/S3 record belongs
to at most one S1 entity:
  per S1   : how this candidate ranks among the S1's candidates
  per cand : how many S1 records compete for this candidate, and whether this
             S1 is the candidate's best-scoring one
"""
import argparse
import os
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from io_utils import log

REC_COLS = [
    "idx", "src", "name_norm", "name_core", "name_compact", "name_tokens", "name_skel", "legal_form",
    "name_nonlatin", "name_website", "addr_norm", "addr_tokens", "addr_numbers", "addr_ids",
    "house_no", "street_key", "addr_empty",
]

# (feature name, column, scorer)
STRING_FEATURES = [
    ("nm_ratio", "name_core", fuzz.ratio),
    ("nm_partial", "name_core", fuzz.partial_ratio),
    ("nm_tsort", "name_core", fuzz.token_sort_ratio),
    ("nm_tset", "name_core", fuzz.token_set_ratio),
    ("nm_wratio", "name_core", fuzz.WRatio),
    ("nm_full_tset", "name_norm", fuzz.token_set_ratio),
    ("nc_ratio", "name_compact", fuzz.ratio),
    ("nc_partial", "name_compact", fuzz.partial_ratio),
    ("nc_jw", "name_compact", JaroWinkler.normalized_similarity),
    ("nk_ratio", "name_skel_str", fuzz.ratio),
    ("ad_ratio", "addr_norm", fuzz.ratio),
    ("ad_tset", "addr_norm", fuzz.token_set_ratio),
    ("ad_partial", "addr_norm", fuzz.partial_ratio),
    ("ad_tsort", "addr_norm", fuzz.token_sort_ratio),
]


def _cpdist(a: list, b: list, scorer) -> np.ndarray:
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _set_feats(prefix: str, a: str, b: str) -> list:
    inter = pl.col(a).list.set_intersection(pl.col(b)).list.len()
    la, lb = pl.col(a).list.len(), pl.col(b).list.len()
    union = la + lb - inter
    return [
        inter.cast(pl.Int16).alias(f"{prefix}_inter"),
        pl.when(union > 0).then(inter / union).otherwise(None).cast(pl.Float32).alias(f"{prefix}_jac"),
        pl.when(la > 0).then(inter / la).otherwise(None).cast(pl.Float32).alias(f"{prefix}_cont1"),
        pl.when(lb > 0).then(inter / lb).otherwise(None).cast(pl.Float32).alias(f"{prefix}_cont2"),
    ]


def pair_features(cand: pl.DataFrame, records: pl.DataFrame) -> pl.DataFrame:
    """cand: s1_idx, cand_idx, blk_* columns. records: REC_COLS (indexed by idx)."""
    rec = records.with_columns(pl.col("name_skel").list.join(" ").alias("name_skel_str"))
    left = rec.rename({c: f"{c}_1" for c in rec.columns if c != "idx"}).rename({"idx": "s1_idx"})
    right = rec.rename({c: f"{c}_2" for c in rec.columns if c != "idx"}).rename({"idx": "cand_idx"})
    df = cand.join(left, on="s1_idx", how="left").join(right, on="cand_idx", how="left")

    feats = {}
    for fname, col, scorer in STRING_FEATURES:
        feats[fname] = _cpdist(df[f"{col}_1"].to_list(), df[f"{col}_2"].to_list(), scorer)
    # house-number edit distance (catches "1630" vs "630", "7704" vs "704")
    feats["house_lev"] = _cpdist(df["house_no_1"].to_list(), df["house_no_2"].to_list(), Levenshtein.distance)
    df = df.with_columns([pl.Series(k, v) for k, v in feats.items()])

    both_house = (pl.col("house_no_1") != "") & (pl.col("house_no_2") != "")
    df = df.with_columns(
        *_set_feats("ntok", "name_tokens_1", "name_tokens_2"),
        *_set_feats("nskel", "name_skel_1", "name_skel_2"),
        *_set_feats("atok", "addr_tokens_1", "addr_tokens_2"),
        *_set_feats("anum", "addr_numbers_1", "addr_numbers_2"),
        pl.col("addr_ids_1").list.set_intersection(pl.col("addr_ids_2")).list.len().cast(pl.Int16).alias("aid_inter"),
        (pl.col("name_compact_1") == pl.col("name_compact_2")).cast(pl.Int8).alias("nc_exact"),
        (pl.col("name_tokens_1").list.first() == pl.col("name_tokens_2").list.first()).cast(pl.Int8).alias("ntok_first_eq"),
        (pl.col("name_compact_2").str.contains(pl.col("name_compact_1"), literal=True)
         | pl.col("name_compact_1").str.contains(pl.col("name_compact_2"), literal=True)).cast(pl.Int8).alias("nc_contains"),
        pl.when(both_house).then((pl.col("house_no_1") == pl.col("house_no_2")).cast(pl.Int8)).otherwise(None).alias("house_eq"),
        pl.when(both_house).then(pl.col("house_lev")).otherwise(None).alias("house_lev"),
        # digit dropped at the front ("5235" vs "235") is a noise pattern on true matches,
        # a different last digit ("3370" vs "3377") usually means a different business
        pl.when(both_house).then(
            (pl.col("house_no_1").str.ends_with(pl.col("house_no_2"))
             | pl.col("house_no_2").str.ends_with(pl.col("house_no_1"))).cast(pl.Int8)
        ).otherwise(None).alias("house_suffix"),
        pl.when(both_house).then(
            (pl.col("house_no_1").str.starts_with(pl.col("house_no_2"))
             | pl.col("house_no_2").str.starts_with(pl.col("house_no_1"))).cast(pl.Int8)
        ).otherwise(None).alias("house_prefix"),
        pl.when(both_house).then(
            (pl.col("house_no_1").str.slice(0, 9).cast(pl.Float64, strict=False)
             - pl.col("house_no_2").str.slice(0, 9).cast(pl.Float64, strict=False)).abs().log1p().cast(pl.Float32)
        ).otherwise(None).alias("house_absdiff"),
        pl.when((pl.col("street_key_1") != "") & (pl.col("street_key_2") != ""))
        .then((pl.col("street_key_1") == pl.col("street_key_2")).cast(pl.Int8)).otherwise(None).alias("street_eq"),
        pl.when((pl.col("legal_form_1") != "") & (pl.col("legal_form_2") != ""))
        .then((pl.col("legal_form_1") == pl.col("legal_form_2")).cast(pl.Int8)).otherwise(None).alias("legal_eq"),
        pl.col("name_tokens_1").list.len().cast(pl.Int16).alias("ntok_len1"),
        pl.col("name_tokens_2").list.len().cast(pl.Int16).alias("ntok_len2"),
        pl.col("name_compact_1").str.len_chars().cast(pl.Int16).alias("nc_len1"),
        pl.col("name_compact_2").str.len_chars().cast(pl.Int16).alias("nc_len2"),
        pl.col("addr_tokens_2").list.len().cast(pl.Int16).alias("atok_len2"),
        pl.col("name_nonlatin_2").cast(pl.Int8).alias("cand_nonlatin"),
        pl.col("name_website_2").cast(pl.Int8).alias("cand_website"),
        pl.col("addr_empty_1").cast(pl.Int8).alias("s1_addr_empty"),
        pl.col("addr_empty_2").cast(pl.Int8).alias("cand_addr_empty"),
        (pl.col("src_2") == 3).cast(pl.Int8).alias("cand_is_s3"),
    )
    # empty address -> address similarities are meaningless, not "0% similar"
    addr_missing = (pl.col("addr_empty_1") | pl.col("addr_empty_2"))
    df = df.with_columns([
        pl.when(addr_missing).then(None).otherwise(pl.col(c)).alias(c)
        for c in ["ad_ratio", "ad_tset", "ad_partial", "ad_tsort", "atok_jac", "atok_cont1", "atok_cont2"]
    ])
    keep = ["s1_idx", "cand_idx"] + [c for c in df.columns if c in FEATURES_BASE or c.startswith("blk_")]
    return df.select(keep)


FEATURES_BASE = (
    [f for f, _, _ in STRING_FEATURES]
    + [f"{p}_{s}" for p in ("ntok", "nskel", "atok", "anum") for s in ("inter", "jac", "cont1", "cont2")]
    + ["aid_inter", "nc_exact", "ntok_first_eq", "nc_contains", "house_eq", "house_lev", "house_suffix",
       "house_prefix", "house_absdiff", "street_eq",
       "legal_eq", "ntok_len1", "ntok_len2", "nc_len1", "nc_len2", "atok_len2", "cand_nonlatin",
       "cand_website", "s1_addr_empty", "cand_addr_empty", "cand_is_s3"]
)


def candidate_context(cand: pl.DataFrame) -> pl.DataFrame:
    """Global per-candidate competition features, from blocking scores over ALL S1."""
    return cand.with_columns(
        pl.len().over("cand_idx").cast(pl.Int16).alias("cx_cand_n_s1"),
        pl.col("blk_score").rank("ordinal", descending=True).over("cand_idx").cast(pl.Int16).alias("cx_cand_rank"),
        (pl.col("blk_score") - pl.col("blk_score").max().over("cand_idx")).alias("cx_cand_gap"),
        pl.len().over("s1_idx").cast(pl.Int16).alias("cx_s1_n_cands"),
    )


S1_CONTEXT_COLS = ["nm_tset", "nc_ratio", "ad_tset", "blk_score"]


def s1_context(df: pl.DataFrame) -> pl.DataFrame:
    """Per-S1 relative features: is this candidate the best-looking one for this S1?"""
    exprs = []
    for c in S1_CONTEXT_COLS:
        exprs += [
            (pl.col(c) - pl.col(c).max().over("s1_idx")).alias(f"cx_{c}_gap"),
            pl.col(c).rank("min", descending=True).over("s1_idx").cast(pl.Int16).alias(f"cx_{c}_rank"),
        ]
    exprs.append((pl.col("nm_tset") >= 90).sum().over("s1_idx").cast(pl.Int16).alias("cx_s1_n_strong_name"))
    exprs.append(((pl.col("nm_tset") >= 90) & (pl.col("ad_tset").fill_null(0) >= 80)).sum().over("s1_idx")
                 .cast(pl.Int16).alias("cx_s1_n_strong_both"))
    return df.with_columns(exprs)


def feature_columns(df: pl.DataFrame) -> list:
    return [c for c in df.columns if c not in ("s1_idx", "cand_idx", "label", "fold")]


def _stage_candidates(d: str, s1_subset) -> str:
    """Global candidate context on a slim 3-column table, joined back to the
    (optionally S1-subsampled) candidates, sorted by s1_idx and spilled to disk
    so feature chunks can be streamed back with row-group pruning."""
    path = os.path.join(d, "candidates.parquet")
    slim = pl.read_parquet(path, columns=["s1_idx", "cand_idx", "blk_score"])
    ctx = candidate_context(slim).drop("blk_score")
    del slim
    lf = pl.scan_parquet(path)
    if s1_subset is not None:
        keep = pl.col("s1_idx").is_in(s1_subset.implode())
        ctx = ctx.filter(keep)
        lf = lf.filter(keep)
    staged = os.path.join(d, "_cand_ctx.parquet")
    lf.collect().join(ctx, on=["s1_idx", "cand_idx"]).sort("s1_idx").write_parquet(staged, row_group_size=500_000)
    return staged


def build_features(work_dir: str, dataset: str, chunk_s1: int = 100_000, s1_subset: pl.Series = None):
    """Writes <work>/<dataset>/features/part-*.parquet (chunked by S1 so memory stays bounded)."""
    d = os.path.join(work_dir, dataset)
    out_dir = os.path.join(d, "features")
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))

    t0 = time.time()
    staged = _stage_candidates(d, s1_subset)
    log(f"candidate context staged ({time.time() - t0:.0f}s)")
    records = pl.read_parquet(os.path.join(d, "records.parquet"), columns=REC_COLS)

    labels = None
    if os.path.exists(os.path.join(d, "gt_pairs.parquet")):
        labels = pl.read_parquet(os.path.join(d, "gt_pairs.parquet")).with_columns(pl.lit(1, pl.Int8).alias("label"))
        meta = pl.read_parquet(os.path.join(d, "s1_meta.parquet")).select("s1_idx", "fold")

    emb = None
    emb_path = os.path.join(d, "emb.npy")
    if os.path.exists(emb_path):
        from embed import pair_cosine
        emb = np.load(emb_path, mmap_mode="r")
        log(f"adding emb_cos feature from {emb_path}")

    s1_ids = pl.scan_parquet(staged).select(pl.col("s1_idx").unique()).collect()["s1_idx"].sort()
    n_parts = 0
    for start in range(0, len(s1_ids), chunk_s1):
        ids = s1_ids.slice(start, chunk_s1)
        lo, hi = ids[0], ids[-1]
        c = pl.scan_parquet(staged).filter(pl.col("s1_idx").is_between(lo, hi)).collect()
        need = pl.concat([c["s1_idx"], c["cand_idx"]]).unique()
        rec = records.filter(pl.col("idx").is_in(need.implode()))
        f = pair_features(c, rec)
        f = f.join(c.select("s1_idx", "cand_idx", *[x for x in c.columns if x.startswith("cx_")]),
                   on=["s1_idx", "cand_idx"])
        del c, rec
        if emb is not None:
            f = f.with_columns(pl.Series("emb_cos", pair_cosine(emb, f["s1_idx"].to_numpy(), f["cand_idx"].to_numpy())))
        f = s1_context(f)
        if labels is not None:
            f = f.join(labels, on=["s1_idx", "cand_idx"], how="left").with_columns(pl.col("label").fill_null(0))
            f = f.join(meta, on="s1_idx", how="left")
        f.write_parquet(os.path.join(out_dir, f"part-{n_parts:04d}.parquet"))
        n_parts += 1
        log(f"  features chunk {n_parts}: {f.height} pairs ({time.time() - t0:.0f}s elapsed)")
        del f
    os.remove(staged)
    log(f"features done: {n_parts} parts in {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--dataset", choices=["train", "test"], required=True)
    ap.add_argument("--chunk-s1", type=int, default=100_000)
    ap.add_argument("--s1-frac", type=float, default=1.0,
                    help="train only: featurize a hash-sample of S1 (laptop memory); context stays global")
    args = ap.parse_args()
    subset = None
    if args.s1_frac < 1.0:
        meta = pl.read_parquet(os.path.join(args.work_dir, args.dataset, "s1_meta.parquet"))
        subset = meta.filter((pl.col("s1_idx").hash(3) % 10_000) < int(args.s1_frac * 10_000))["s1_idx"]
    build_features(args.work_dir, args.dataset, args.chunk_s1, subset)


if __name__ == "__main__":
    main()
