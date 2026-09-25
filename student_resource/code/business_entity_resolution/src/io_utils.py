"""Reading challenge TSVs and writing submission TSVs."""
import hashlib
import json
import os
import time

import polars as pl

TSV_OPTS = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
SRC_CODE = {"S1": 1, "S2": 2, "S3": 3}


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def source_paths(data_dir: str, prefix: str):
    return [os.path.join(data_dir, f"{prefix}_source{i}.tsv") for i in (1, 2, 3)]


def read_sources(data_dir: str, prefix: str) -> pl.DataFrame:
    """All three sources stacked, with `src` (1/2/3) and a dense `idx`."""
    frames = []
    for i, path in enumerate(source_paths(data_dir, prefix), start=1):
        df = pl.read_csv(path, **TSV_OPTS).with_columns(pl.lit(i, dtype=pl.Int8).alias("src"))
        frames.append(df)
    df = pl.concat(frames)
    return df.with_row_index("idx").with_columns(pl.col("idx").cast(pl.Int32))


def read_ground_truth_pairs(path: str) -> pl.DataFrame:
    """(s1_id, cand_id) rows for every true match; singletons omitted."""
    gt = pl.read_csv(path, **TSV_OPTS)
    return (
        gt.rename({"source1_entity_id": "s1_id", "matched_entity_ids": "cand_id"})
        .with_columns(pl.col("cand_id").str.split(","))
        .explode("cand_id")
        .filter(pl.col("cand_id").is_not_null() & (pl.col("cand_id") != ""))
    )


def fold_of(entity_id_col: pl.Expr, val_frac: float, seed: int = 13) -> pl.Expr:
    """Deterministic fit/val assignment by hashing the Source-1 id."""
    return pl.when((entity_id_col.hash(seed) % 10_000) < int(val_frac * 10_000)).then(pl.lit("val")).otherwise(pl.lit("fit"))


def write_id_lists(s1_ids: list, groups: pl.DataFrame, id_col: str, out_path: str, header_col: str):
    """One row per S1 id (in the given order); comma-joined ids, empty if none.

    `groups` has columns s1_id and `id_col` (one row per (s1, id) pair).
    """
    agg = groups.group_by("s1_id").agg(pl.col(id_col).unique(maintain_order=True).str.join(",").alias(header_col))
    out = pl.DataFrame({"s1_id": s1_ids}).join(agg, on="s1_id", how="left").with_columns(
        pl.col(header_col).fill_null("")
    )
    out = out.rename({"s1_id": "source1_entity_id"})
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out.write_csv(out_path, separator="\t", quote_style="never")


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def file_digest(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
    return h.hexdigest()[:10]
