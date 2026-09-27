"""End-to-end orchestrator.

  python src/run.py train   --data-dir <student_resource>/dataset --work-dir work --model-dir models
  python src/run.py predict --data-dir <student_resource>/dataset --work-dir work --model-dir models --output-dir output
  python src/run.py all     (both, in order)

train:   prepare(train) -> blocking(train) -> features(train) -> LightGBM -> tune decision rule on val
predict: prepare(test)  -> blocking(test)  -> features(test)  -> score -> decide -> write both TSVs

Every stage caches its output under --work-dir; pass --skip-existing to reuse
cached stages while iterating on a later one.
"""
import argparse
import json
import os
import subprocess
import sys

import polars as pl

import blocking
import decide
import features
import prepare
import train as train_mod
from io_utils import log, write_id_lists

HERE = os.path.dirname(os.path.abspath(__file__))


def _exists(path):
    return os.path.exists(path)


def run_train(args):
    d = os.path.join(args.work_dir, "train")
    if not (args.skip_existing and _exists(os.path.join(d, "records.parquet"))):
        prepare.prepare_train(os.path.join(args.data_dir, "train"), d, args.val_frac, args.sample_frac)
    if not (args.skip_existing and _exists(os.path.join(d, "candidates.parquet"))):
        blocking.generate_candidates(
            os.path.join(d, "records.parquet"), os.path.join(d, "candidates.parquet"),
            k_name=args.k_name, k_addr=args.k_addr, k_both=args.k_both, max_cands=args.max_cands,
            max_df_frac=args.max_df_frac, n_threads=args.threads, extra=blocking.load_extra(d))
    cand = pl.read_parquet(os.path.join(d, "candidates.parquet"), columns=["s1_idx", "cand_idx", "blk_rank"])
    gt = pl.read_parquet(os.path.join(d, "gt_pairs.parquet"))
    meta = pl.read_parquet(os.path.join(d, "s1_meta.parquet"))
    rep = blocking.blocking_report(cand, gt, meta)
    log(f"blocking report (all train S1): {rep}")
    del cand

    if not (args.skip_existing and _exists(os.path.join(d, "features"))):
        subset = None
        if args.feature_s1_frac < 1.0:
            subset = meta.filter((pl.col("s1_idx").hash(3) % 10_000) < int(args.feature_s1_frac * 10_000))["s1_idx"]
        features.build_features(args.work_dir, "train", args.chunk_s1, subset)

    if not (args.skip_existing and _exists(os.path.join(args.model_dir, "model.txt"))):
        train_mod.train(args.work_dir, args.model_dir, args.neg_frac, args.rounds, args.early_stop,
                        {"learning_rate": args.learning_rate, "num_threads": args.threads})

    scored_val = train_mod.predict(args.work_dir, "train", args.model_dir, folds=["val"])
    # every val entity counts, including ones blocking found no candidates for
    meta_val = meta.filter(pl.col("fold") == "val")
    if args.feature_s1_frac < 1.0:
        meta_val = meta_val.filter((pl.col("s1_idx").hash(3) % 10_000) < int(args.feature_s1_frac * 10_000))
    best, best_f, _ = decide.tune(scored_val, meta_val)
    pred = decide.apply_rule(scored_val, best)
    report = {"val_macro_f05": round(best_f, 4), "decision": best, "blocking": rep,
              **decide.breakdown(pred, meta_val)}
    with open(os.path.join(args.model_dir, "decision.json"), "w") as f:
        json.dump(best, f, indent=2)
    with open(os.path.join(args.model_dir, "val_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"VALIDATION REPORT: {json.dumps(report)}")


def run_predict(args):
    d = os.path.join(args.work_dir, "test")
    if not (args.skip_existing and _exists(os.path.join(d, "records.parquet"))):
        prepare.prepare_test(os.path.join(args.data_dir, "test"), d,
                             os.path.join(args.work_dir, "train", "translit.json"))
    if not (args.skip_existing and _exists(os.path.join(d, "candidates.parquet"))):
        blocking.generate_candidates(
            os.path.join(d, "records.parquet"), os.path.join(d, "candidates.parquet"),
            k_name=args.k_name, k_addr=args.k_addr, k_both=args.k_both, max_cands=args.max_cands,
            max_df_frac=args.max_df_frac, n_threads=args.threads, extra=blocking.load_extra(d))
    if not (args.skip_existing and _exists(os.path.join(d, "features"))):
        features.build_features(args.work_dir, "test", args.chunk_s1)

    scored = train_mod.predict(args.work_dir, "test", args.model_dir)
    with open(os.path.join(args.model_dir, "decision.json")) as f:
        params = json.load(f)
    pred = decide.apply_rule(scored, params)
    log(f"decision {params}: {pred.height} matched pairs for {pred['s1_idx'].n_unique()} S1 entities")

    ids = pl.read_parquet(os.path.join(d, "records.parquet"), columns=["idx", "entity_id", "src"])
    s1_order = ids.filter(pl.col("src") == 1)["entity_id"].to_list()
    id_map = ids.select("idx", "entity_id")

    def to_ids(pairs):
        return (pairs.join(id_map.rename({"idx": "s1_idx", "entity_id": "s1_id"}), on="s1_idx", maintain_order="left")
                .join(id_map.rename({"idx": "cand_idx", "entity_id": "cand_id"}), on="cand_idx", maintain_order="left")
                .select("s1_id", "cand_id"))

    cand = pl.read_parquet(os.path.join(d, "candidates.parquet"), columns=["s1_idx", "cand_idx", "blk_rank"])
    cand = cand.sort(["s1_idx", "blk_rank"])
    os.makedirs(args.output_dir, exist_ok=True)
    write_id_lists(s1_order, to_ids(cand), "cand_id", os.path.join(args.output_dir, "candidate_pairs.tsv"),
                   "candidate_entity_ids")
    write_id_lists(s1_order, to_ids(pred.sort(["s1_idx", "p"], descending=[False, True])), "cand_id",
                   os.path.join(args.output_dir, "matching_results.tsv"), "matched_entity_ids")
    log(f"wrote {args.output_dir}/matching_results.tsv and candidate_pairs.tsv")

    validator = os.path.join(args.data_dir, "..", "utils", "validate_submission.py")
    if os.path.exists(validator):
        subprocess.run([sys.executable, validator,
                        "--matching", os.path.join(args.output_dir, "matching_results.tsv"),
                        "--candidate", os.path.join(args.output_dir, "candidate_pairs.tsv"),
                        "--test-dir", os.path.join(args.data_dir, "test")], check=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "predict", "all"])
    ap.add_argument("--data-dir", required=True, help="folder containing train/ and test/")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--skip-existing", action="store_true", help="reuse cached stage outputs")
    # data
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--sample-frac", type=float, default=1.0, help="train mini-world fraction (laptop dev)")
    ap.add_argument("--feature-s1-frac", type=float, default=1.0, help="featurize only this fraction of train S1")
    # blocking
    ap.add_argument("--k-name", type=int, default=30)
    ap.add_argument("--k-addr", type=int, default=30)
    ap.add_argument("--k-both", type=int, default=40)
    ap.add_argument("--max-cands", type=int, default=100)
    ap.add_argument("--max-df-frac", type=float, default=0.002)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--chunk-s1", type=int, default=100_000)
    # model
    ap.add_argument("--neg-frac", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--early-stop", type=int, default=100)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    args = ap.parse_args()

    if args.cmd in ("train", "all"):
        run_train(args)
    if args.cmd in ("predict", "all"):
        run_predict(args)


if __name__ == "__main__":
    main()
