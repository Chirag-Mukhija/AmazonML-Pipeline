"""Error analysis on the validation split.

Writes <model_dir>/errors/{false_positives,false_negatives}.tsv with the raw
names/addresses side by side, and prints macro F0.5 per country and per
source. This is the fastest way for the team to find what to fix next.

  python src/analyze.py --work-dir work --model-dir models
"""
import argparse
import json
import os

import polars as pl

import decide
import train as train_mod
from io_utils import log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--n", type=int, default=500, help="examples per error file")
    args = ap.parse_args()

    d = os.path.join(args.work_dir, "train")
    meta = pl.read_parquet(os.path.join(d, "s1_meta.parquet")).filter(pl.col("fold") == "val")
    gt = pl.read_parquet(os.path.join(d, "gt_pairs.parquet"))
    rec = pl.read_parquet(os.path.join(d, "records.parquet"),
                          columns=["idx", "entity_id", "business_name", "business_address", "country"])
    with open(os.path.join(args.model_dir, "decision.json")) as f:
        params = json.load(f)

    scored = train_mod.predict(args.work_dir, "train", args.model_dir, folds=["val"])
    pred = decide.apply_rule(scored, params)

    country = rec.select(pl.col("idx").alias("s1_idx"), "country")
    for c in meta.join(country, on="s1_idx")["country"].unique().to_list():
        mc = meta.join(country, on="s1_idx").filter(pl.col("country") == c).drop("country")
        pc = pred.filter(pl.col("s1_idx").is_in(mc["s1_idx"].implode()))
        log(f"country={c}: macro F0.5={decide.macro_f05(pc, mc):.4f} n={mc.height} {decide.breakdown(pc, mc)}")

    def side(df):
        a = rec.rename({"idx": "s1_idx", "entity_id": "s1_id", "business_name": "s1_name",
                        "business_address": "s1_addr"}).drop("country")
        b = rec.rename({"idx": "cand_idx", "entity_id": "cand_id", "business_name": "cand_name",
                        "business_address": "cand_addr"})
        return df.join(a, on="s1_idx").join(b, on="cand_idx")

    out = os.path.join(args.model_dir, "errors")
    os.makedirs(out, exist_ok=True)
    fp = side(pred.filter(pl.col("label") == 0).sort("p", descending=True).head(args.n))
    val_gt = gt.filter(pl.col("s1_idx").is_in(meta["s1_idx"].implode()))
    fn = val_gt.join(pred.select("s1_idx", "cand_idx"), on=["s1_idx", "cand_idx"], how="anti")
    fn = fn.join(scored.select("s1_idx", "cand_idx", "p"), on=["s1_idx", "cand_idx"], how="left")
    log(f"false negatives: {fn.height} ({fn['p'].is_null().sum()} never reached the model = blocking misses)")
    fn = side(fn.sample(min(args.n, fn.height), seed=0))
    fp.write_csv(os.path.join(out, "false_positives.tsv"), separator="\t")
    fn.write_csv(os.path.join(out, "false_negatives.tsv"), separator="\t")
    log(f"wrote error samples to {out}")


if __name__ == "__main__":
    main()
