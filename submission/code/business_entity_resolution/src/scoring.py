"""Local F_0.5 scorer that mirrors the leaderboard formula exactly.

F_0.5 = (1.25 * P * R) / (0.25 * P + R), computed per Source-1 entity and
macro-averaged across every entity in the evaluation set. A singleton with
no true matches scores 1.0 for an empty prediction and 0.0 for any non-empty
prediction (P is undefined there, so we special-case it as the spec does).

Usage:
    python3 src/scoring.py --pred output/matching_results.tsv --truth <ground_truth.tsv>
"""
import argparse
import csv
import sys

DELIM = "\t"


def read_id_lists(path: str) -> dict:
    """source1_entity_id -> set(matched ids), from a 2-col TSV with a header."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=DELIM)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            if len(row) == 1:
                s1_id, ids = row[0], ""
            else:
                s1_id, ids = row[0], row[1]
            ids = {x for x in ids.split(",") if x} if ids else set()
            out[s1_id] = ids
    return out


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def score_entity(pred: set, truth: set) -> float:
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    precision = tp / len(pred)
    recall = tp / len(truth)
    return f_beta(precision, recall, beta=0.5)


def score(pred_path: str, truth_path: str, verbose: bool = True) -> float:
    preds = read_id_lists(pred_path)
    truths = read_id_lists(truth_path)

    missing = set(truths) - set(preds)
    if missing:
        raise ValueError(
            f"{len(missing)} source1 entities from truth are missing in predictions "
            f"(e.g. {sorted(missing)[:5]}) -- every entity must have a row."
        )

    scores = [score_entity(preds[s1], truth) for s1, truth in truths.items()]
    macro_f05 = sum(scores) / len(scores) if scores else 0.0

    if verbose:
        n_singletons = sum(1 for t in truths.values() if not t)
        n_singleton_correct = sum(
            1 for s1, t in truths.items() if not t and not preds[s1]
        )
        print(f"Entities scored:       {len(scores)}")
        print(f"Macro F_0.5:            {macro_f05:.4f}")
        print(f"Singletons in truth:    {n_singletons}")
        print(f"Singletons predicted correctly: {n_singleton_correct}/{n_singletons}")

    return macro_f05


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True, help="matching_results.tsv (or a validation-split prediction)")
    ap.add_argument("--truth", required=True, help="ground truth TSV with the same 2-column schema")
    args = ap.parse_args()
    try:
        score(args.pred, args.truth)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
