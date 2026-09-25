"""Stage 0: load raw TSVs, split train into fit/val, learn the transliteration
dictionary (fit split only), normalize every record, cache to parquet.

Outputs in <work_dir>/<dataset>/:
  records.parquet   one row per record of all 3 sources, normalized
  gt_pairs.parquet  (train only) s1_idx, cand_idx for every true match
  s1_meta.parquet   (train only) s1_idx, fold ("fit"/"val"), n_true
  translit.json     (train only; reused for test)

`--sample-frac` builds a self-consistent "mini-world" from train for fast
laptop iteration: a hash-sample of Source-1 entities + ALL of their true
matches + the same fraction of unmatched Source-2/3 distractors, so match
structure (and the ~26% distractor rate) is preserved at smaller scale.
"""
import argparse
import os

import polars as pl

from io_utils import fold_of, load_json, log, read_ground_truth_pairs, read_sources, save_json
from normalize import learn_translit_dict, normalize_frame


def prepare_train(data_dir: str, out_dir: str, val_frac: float, sample_frac: float):
    os.makedirs(out_dir, exist_ok=True)
    log("reading train sources")
    records = read_sources(data_dir, "train").drop("idx")
    gt = read_ground_truth_pairs(os.path.join(data_dir, "train_ground_truth.tsv"))

    all_matched = gt["cand_id"]
    if sample_frac < 1.0:
        s1 = records.filter((pl.col("src") == 1) & ((pl.col("entity_id").hash(7) % 10_000) < int(sample_frac * 10_000)))
        gt = gt.filter(pl.col("s1_id").is_in(s1["entity_id"].implode()))
        pool = records.filter(pl.col("src") != 1)
        is_matched_anywhere = pl.col("entity_id").is_in(all_matched.implode())
        pool = pool.filter(
            pl.col("entity_id").is_in(gt["cand_id"].implode())
            | (~is_matched_anywhere & ((pl.col("entity_id").hash(11) % 10_000) < int(sample_frac * 10_000)))
        )
        records = pl.concat([s1, pool])
        log(f"mini-world: {s1.height} S1, {pool.height} S2/S3 records, {gt.height} true pairs")

    records = records.with_row_index("idx").with_columns(pl.col("idx").cast(pl.Int32))

    s1 = records.filter(pl.col("src") == 1).select("idx", "entity_id")
    s1_meta = s1.with_columns(fold_of(pl.col("entity_id"), val_frac).alias("fold"))

    id2idx = records.select("entity_id", "idx")
    gt_pairs = (
        gt.join(id2idx.rename({"entity_id": "s1_id", "idx": "s1_idx"}), on="s1_id")
        .join(id2idx.rename({"entity_id": "cand_id", "idx": "cand_idx"}), on="cand_id")
        .select("s1_idx", "cand_idx")
    )
    n_true = gt_pairs.group_by("s1_idx").len().rename({"len": "n_true"})
    s1_meta = s1_meta.join(n_true, left_on="idx", right_on="s1_idx", how="left").with_columns(
        pl.col("n_true").fill_null(0)
    ).rename({"idx": "s1_idx"}).drop("entity_id")

    log("learning transliteration dictionary from FIT-split pairs only")
    fit_ids = s1_meta.filter(pl.col("fold") == "fit")["s1_idx"]
    raw = records.select("idx", "business_name")
    fit_pairs = (
        gt_pairs.filter(pl.col("s1_idx").is_in(fit_ids.implode()))
        .join(raw.rename({"idx": "s1_idx", "business_name": "n1"}), on="s1_idx")
        .join(raw.rename({"idx": "cand_idx", "business_name": "n2"}), on="cand_idx")
    )
    translit = learn_translit_dict(fit_pairs.select("n1", "n2"))
    save_json(translit, os.path.join(out_dir, "translit.json"))
    log(f"translit entries: {len(translit)}")

    log(f"normalizing {records.height} records")
    records = normalize_frame(records, translit)
    records.write_parquet(os.path.join(out_dir, "records.parquet"))
    gt_pairs.write_parquet(os.path.join(out_dir, "gt_pairs.parquet"))
    s1_meta.write_parquet(os.path.join(out_dir, "s1_meta.parquet"))
    log(f"train prepared: {s1_meta.height} S1 ({(s1_meta['fold'] == 'val').sum()} val), "
        f"{gt_pairs.height} true pairs")


def prepare_test(data_dir: str, out_dir: str, translit_path: str):
    os.makedirs(out_dir, exist_ok=True)
    log("reading test sources")
    records = read_sources(data_dir, "test")
    translit = load_json(translit_path)
    log(f"normalizing {records.height} records")
    records = normalize_frame(records, translit)
    records.write_parquet(os.path.join(out_dir, "records.parquet"))
    log("test prepared")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["train", "test"], required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--translit", help="translit.json from the train run (test only)")
    args = ap.parse_args()
    out = os.path.join(args.work_dir, args.dataset)
    if args.dataset == "train":
        prepare_train(args.data_dir, out, args.val_frac, args.sample_frac)
    else:
        prepare_test(args.data_dir, out, args.translit or os.path.join(args.work_dir, "train", "translit.json"))


if __name__ == "__main__":
    main()
