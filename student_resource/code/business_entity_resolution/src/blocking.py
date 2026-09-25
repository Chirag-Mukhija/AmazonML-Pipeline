"""Stage 1: candidate generation (blocking).

Idea: every record emits a set of "blocking keys". Two records become
candidates when they share rare keys. Implemented as IDF-weighted sparse
matrices and a multithreaded top-K sparse product (sparse_dot_topn), per
country, so it scales to millions x millions without all-pairs comparison.

  name keys     t:<token>  k:<consonant skeleton>  c:<compact name>
                p:<first 6 chars of compact name>  (catches "acmecorp.com")
  address keys  s:<house#_street>  i:<alnum id, e.g. b112>  n:<number, 3+ digits>
                b:<address token bigram>

Keys whose document frequency in the S2/S3 pool exceeds `max_df_frac` of the
pool are dropped (uninformative, and they dominate cost). Three channels each
keep their own top-K per S1 record: "name" (name keys only), "addr" (address
keys only) and "both" (score = name + address). The union -- NOT re-ranked
across channels -- is exactly what the matcher sees, i.e. candidate_pairs.tsv.

`--s1-frac 0.05` blocks a hash-sample of S1 against the full pool: per-S1
results are identical to a full run, so it's a fast exact recall estimate.
"""
import argparse
import os
import shutil
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

from io_utils import log

BLOCKING_COLUMNS = ["idx", "src", "country", "name_tokens", "name_skel", "name_compact", "street_key",
                    "addr_ids", "addr_numbers", "addr_tokens"]


def record_keys(records: pl.DataFrame) -> pl.DataFrame:
    """Long table: idx, channel ("name"/"addr"), key (u64 hash of prefix+key).

    Call per country: keys are only ever compared within one country.
    """
    r = records.select(
        "idx", "name_tokens", "name_skel", "name_compact", "street_key",
        "addr_ids", "addr_numbers", "addr_tokens",
    )
    parts = []

    def add(expr_list_col, prefix, channel, min_len=2):
        parts.append(
            r.select("idx", expr_list_col.alias("k"))
            .explode("k")
            .filter(pl.col("k").is_not_null() & (pl.col("k").str.len_chars() >= min_len))
            .select(
                "idx",
                (pl.lit(prefix) + pl.col("k")).hash(42).alias("key"),
                pl.lit(channel, dtype=pl.Categorical).alias("channel"),
            )
        )

    add(pl.col("name_tokens"), "t:", "name")
    add(pl.col("name_skel"), "k:", "name")
    add(pl.concat_list(pl.col("name_compact")), "c:", "name", min_len=4)
    add(pl.concat_list(pl.col("name_compact").str.slice(0, 6)), "p:", "name", min_len=6)
    add(pl.concat_list(pl.col("street_key")), "s:", "addr", min_len=3)
    add(pl.col("addr_ids"), "i:", "addr")
    add(pl.col("addr_numbers"), "n:", "addr", min_len=3)
    bigrams = pl.col("addr_tokens").list.eval(
        pl.element() + "_" + pl.element().shift(-1)
    ).list.eval(pl.element().filter(pl.element().is_not_null() & ~pl.element().str.contains(r"^\d+_|_\d+$")))
    add(bigrams, "b:", "addr", min_len=5)

    return pl.concat(parts).unique(["idx", "key"])


def _channel_matrices(s1_keys, pool_keys, n_pool_total, max_df_frac):
    """Build IDF-weighted query matrix Q (s1 x key) and posting matrix PT (key x pool)."""
    df = pool_keys.group_by("key").len().rename({"len": "df"})
    max_df = max(20, int(max_df_frac * n_pool_total))
    df = df.filter(pl.col("df") <= max_df)
    df = df.join(s1_keys.select("key").unique(), on="key", how="semi")
    df = df.with_row_index("kid").with_columns(
        pl.col("kid").cast(pl.Int32),
        (1.0 + (n_pool_total / pl.col("df")).log()).cast(pl.Float32).alias("w"),
    )
    q = s1_keys.join(df.select("key", "kid", "w"), on="key")
    p = pool_keys.join(df.select("key", "kid"), on="key")
    return df.height, q, p, max_df


def _to_csr(rows, cols, vals, shape):
    return sp.csr_matrix((vals, (rows, cols)), shape=shape, dtype=np.float32)


def _pair_dot(Q: sp.csr_matrix, P: sp.csr_matrix, qi: np.ndarray, pi: np.ndarray, chunk=1_000_000) -> np.ndarray:
    """Row-aligned dot products Q[qi[j]] . P[pi[j]] without materializing Q P^T."""
    out = np.empty(len(qi), dtype=np.float32)
    for s in range(0, len(qi), chunk):
        a = Q[qi[s:s + chunk]]
        b = P[pi[s:s + chunk]]
        out[s:s + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


def _channel(keys_c, s1, pool, channel, n_pool, max_df_frac):
    ck = keys_c.filter(pl.col("channel") == channel)
    s1_keys = ck.join(s1, on="idx").select("qrow", "key")
    pool_keys = ck.join(pool, on="idx").select("prow", "key")
    del ck
    n_keys, q, p, max_df = _channel_matrices(s1_keys, pool_keys, n_pool, max_df_frac)
    del s1_keys, pool_keys
    Q = _to_csr(q["qrow"].to_numpy(), q["kid"].to_numpy(), q["w"].to_numpy(), (s1.height, n_keys))
    P = _to_csr(p["prow"].to_numpy(), p["kid"].to_numpy(), np.ones(p.height, np.float32), (n_pool, n_keys))
    del q, p
    # normalize query rows so scores = weighted fraction of the S1 record's keys found
    norm = np.asarray(Q.sum(axis=1)).ravel()
    norm[norm == 0] = 1.0
    Q = (sp.diags(1.0 / norm).astype(np.float32) @ Q).tocsr()
    log(f"    {channel}: {n_keys} keys (max_df={max_df})")
    return Q, P


def _topn(Q, PT, k, channel, n_threads):
    R = sp_matmul_topn(Q, PT, top_n=k, threshold=1e-6, n_threads=n_threads).tocoo()
    f = pl.DataFrame({"qrow": R.row.astype(np.int32), "prow": R.col.astype(np.int32),
                      f"_s_{channel}": R.data.astype(np.float32)})
    del R
    return f.with_columns(pl.col(f"_s_{channel}").rank("ordinal", descending=True).over("qrow")
                          .cast(pl.Int16).alias(f"blk_rank_{channel}"))


def _union_chunk(found, mats, ks, max_cands, ex):
    """Union one S1-chunk's channel hits, fill missing channel scores, rank."""
    cand = found[0]
    for f in found[1:]:
        cand = cand.join(f, on=["qrow", "prow"], how="full", coalesce=True)
    cand = cand.with_columns(pl.lit(False).alias("from_extra"))
    if ex is not None and ex.height:
        cand = cand.join(ex, on=["qrow", "prow"], how="full", coalesce=True).with_columns(
            (pl.col("from_extra").fill_null(False) | pl.col("from_extra_right").fill_null(False)).alias("from_extra")
        ).drop("from_extra_right")
    # name/addr scores for every pair: reuse the channel's own score where it has one
    for channel in ("name", "addr"):
        Q, P = mats[channel]
        col, src_col = f"blk_{channel}", f"_s_{channel}"
        cand = cand.with_columns(pl.col(src_col).alias(col) if src_col in cand.columns
                                 else pl.lit(None, pl.Float32).alias(col))
        miss = cand[col].is_null()
        if miss.any():
            sub = cand.filter(miss)
            vals = _pair_dot(Q, P, sub["qrow"].to_numpy(), sub["prow"].to_numpy())
            cand = cand.with_columns(cand[col].scatter(miss.arg_true(), vals).alias(col))
    cand = cand.with_columns((pl.col("blk_name") + pl.col("blk_addr")).alias("blk_score"))
    cand = cand.with_columns(
        pl.col("blk_score").rank("ordinal", descending=True).over("qrow").alias("blk_rank"),
        *[pl.col(f"blk_rank_{c}").fill_null(k + 1) for c, k in ks.items() if k > 0],
    )
    if max_cands > 0:  # safety cap only; channel top-Ks already bound the size
        cand = cand.filter((pl.col("blk_rank") <= max_cands) | pl.col("from_extra"))
    return cand


def block_country(records_c: pl.DataFrame, keys_c: pl.DataFrame, ks: dict, max_cands: int,
                  max_df_frac: float, n_threads: int, extra: pl.DataFrame = None,
                  chunk_s1: int = 250_000):
    """Yields candidate DataFrames, one per chunk of S1 records.

    ks: top-K per channel, e.g. {"name": 30, "addr": 30, "both": 40}. Channels
    keep their own top-K and are unioned WITHOUT re-ranking across channels, so
    a strong-name / empty-address match is not crowded out by address-similar
    neighbours. "both" ranks by name+address score together.

    Pool-side matrices are built once per country; S1 is processed in chunks,
    so peak memory is bounded regardless of how many S1 records there are.
    """
    s1 = records_c.filter(pl.col("src") == 1).select("idx").with_row_index("qrow")
    pool = records_c.filter(pl.col("src") != 1).select("idx").with_row_index("prow")
    n_pool = pool.height
    if s1.height == 0 or n_pool == 0:
        return

    Qn, Pn = _channel(keys_c, s1, pool, "name", n_pool, max_df_frac)
    Qa, Pa = _channel(keys_c, s1, pool, "addr", n_pool, max_df_frac)
    del keys_c
    mats = {"name": (Qn, Pn), "addr": (Qa, Pa)}
    PT = {"name": Pn.T.tocsr(), "addr": Pa.T.tocsr()}
    Qs = {"name": Qn, "addr": Qa}
    if ks.get("both", 0) > 0:
        # stack existing transposes instead of transposing a new hstack (memory)
        PT["both"] = sp.vstack([PT["name"], PT["addr"]]).tocsr()
        Qs["both"] = sp.hstack([Qn, Qa]).tocsr()
    for c in ("name", "addr"):
        if ks.get(c, 0) <= 0:
            del PT[c]

    ex_all = None
    if extra is not None and extra.height:
        ex_all = (extra.join(s1.rename({"idx": "s1_idx"}), on="s1_idx")
                  .join(pool.rename({"idx": "cand_idx"}), on="cand_idx")
                  .select("qrow", "prow", pl.lit(True).alias("from_extra")))

    for lo in range(0, s1.height, chunk_s1):
        hi = min(lo + chunk_s1, s1.height)
        t = time.time()
        found = []
        for channel, k in ks.items():
            if k <= 0:
                continue
            f = _topn(Qs[channel][lo:hi], PT[channel], k, channel, n_threads)
            found.append(f.with_columns(pl.col("qrow") + lo))
        ex = None if ex_all is None else ex_all.filter(pl.col("qrow").is_between(lo, hi - 1))
        cand = _union_chunk(found, mats, ks, max_cands, ex)
        del found
        out = (
            cand.join(s1.rename({"idx": "s1_idx"}), on="qrow")
            .join(pool.rename({"idx": "cand_idx"}), on="prow")
            .select("s1_idx", "cand_idx", "blk_name", "blk_addr", "blk_score", pl.col("blk_rank").cast(pl.Int16),
                    *[f"blk_rank_{c}" for c, k in ks.items() if k > 0])
        )
        del cand
        log(f"    S1 rows {lo}-{hi}: {out.height} pairs, {time.time() - t:.1f}s")
        yield out


def _country_frame(records, country: str) -> pl.DataFrame:
    """records: a DataFrame, or a path to records.parquet (loaded one country at a time)."""
    if isinstance(records, str):
        return (pl.scan_parquet(records).filter(pl.col("country") == country)
                .select(BLOCKING_COLUMNS).collect())
    return records.filter(pl.col("country") == country)


def _countries(records) -> list:
    if isinstance(records, str):
        return pl.scan_parquet(records).select("country").unique().collect()["country"].sort().to_list()
    return records["country"].unique().sort().to_list()


def generate_candidates(records, out_path: str, k_name=30, k_addr=30, k_both=40, max_cands=100,
                        max_df_frac=0.002, n_threads=8, extra: pl.DataFrame = None,
                        s1_frac: float = 1.0, chunk_s1: int = 250_000) -> str:
    """Writes candidates to `out_path` (parquet), one country / S1-chunk at a time.

    `records`: records.parquet path (memory-friendly) or a DataFrame.
    `extra`: optional (s1_idx, cand_idx) pairs from another channel (embeddings).
    `s1_frac` < 1: block only a hash-sample of S1 against the FULL pool -- per-S1
    results are identical to a full run, so this is a fast, exact recall estimate.
    """
    t0 = time.time()
    ks = {"name": k_name, "addr": k_addr, "both": k_both}
    tmp = out_path + ".parts"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    parts = []
    # Country is an open set: iterate over whatever labels appear (US/India/France/...).
    for country in _countries(records):
        sub = _country_frame(records, country)
        if s1_frac < 1.0:
            sub = sub.filter((pl.col("src") != 1) | ((pl.col("idx").hash(5) % 10_000) < int(s1_frac * 10_000)))
        rc = sub.select("idx", "src")
        keys = record_keys(sub)
        del sub
        log(f"  country={country!r}: {rc.filter(pl.col('src') == 1).height} S1 vs "
            f"{rc.filter(pl.col('src') != 1).height} pool, {keys.height} keys")
        ex = None
        if extra is not None:
            ex = extra.join(rc.filter(pl.col("src") == 1).select(pl.col("idx").alias("s1_idx")), on="s1_idx")
        gen = block_country(rc, keys, ks, max_cands, max_df_frac, n_threads, ex, chunk_s1)
        del keys
        for res in gen:
            p = os.path.join(tmp, f"part-{len(parts):04d}.parquet")
            res.write_parquet(p)
            parts.append(p)
            del res
    pl.scan_parquet(parts).sink_parquet(out_path)
    shutil.rmtree(tmp, ignore_errors=True)
    n = pl.scan_parquet(out_path).select(pl.len()).collect().item()
    log(f"candidates: {n} pairs -> {out_path} ({time.time() - t0:.1f}s)")
    return out_path


def load_extra(dataset_dir: str):
    """Embedding-channel candidates from embed.py, if they were generated."""
    path = os.path.join(dataset_dir, "emb_candidates.parquet")
    if os.path.exists(path):
        log(f"using extra candidates from {path}")
        return pl.read_parquet(path, columns=["s1_idx", "cand_idx"])
    return None


def blocking_report(cand: pl.DataFrame, gt_pairs: pl.DataFrame, s1_meta: pl.DataFrame) -> dict:
    """Pair recall, per-entity recall ceiling, and candidate-set size."""
    n_s1 = s1_meta.height
    hit = gt_pairs.join(cand.select("s1_idx", "cand_idx"), on=["s1_idx", "cand_idx"], how="semi")
    pair_recall = hit.height / max(1, gt_pairs.height)
    per = s1_meta.join(hit.group_by("s1_idx").len().rename({"len": "hit"}), on="s1_idx", how="left").with_columns(
        pl.col("hit").fill_null(0)
    )
    # best achievable macro F0.5 if the matcher were perfect on this candidate set
    per = per.with_columns(
        pl.when(pl.col("n_true") == 0).then(1.0)
        .when(pl.col("hit") == 0).then(0.0)
        .otherwise((1.25 * pl.col("hit") / pl.col("n_true")) / (0.25 + pl.col("hit") / pl.col("n_true")))
        .alias("f_ceiling")
    )
    rep = {
        "pair_recall": round(pair_recall, 4),
        "f05_ceiling": round(per["f_ceiling"].mean(), 4),
        "avg_cands_per_s1": round(cand.height / max(1, n_s1), 1),
        "n_pairs": cand.height,
    }
    for k in (5, 10, 20, 40):
        rep[f"pair_recall@{k}"] = round(
            gt_pairs.join(cand.filter(pl.col("blk_rank") <= k).select("s1_idx", "cand_idx"),
                          on=["s1_idx", "cand_idx"], how="semi").height / max(1, gt_pairs.height), 4)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--dataset", choices=["train", "test"], required=True)
    ap.add_argument("--k-name", type=int, default=30)
    ap.add_argument("--k-addr", type=int, default=30)
    ap.add_argument("--k-both", type=int, default=40)
    ap.add_argument("--max-cands", type=int, default=100)
    ap.add_argument("--max-df-frac", type=float, default=0.002)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--s1-frac", type=float, default=1.0,
                    help="evaluate on a hash-sample of S1 (writes candidates_eval.parquet)")
    ap.add_argument("--chunk-s1", type=int, default=250_000, help="S1 rows per chunk (bounds memory)")
    args = ap.parse_args()

    d = os.path.join(args.work_dir, args.dataset)
    out = os.path.join(d, "candidates.parquet" if args.s1_frac >= 1.0 else "candidates_eval.parquet")
    generate_candidates(os.path.join(d, "records.parquet"), out, k_name=args.k_name, k_addr=args.k_addr,
                        k_both=args.k_both, max_cands=args.max_cands, max_df_frac=args.max_df_frac,
                        n_threads=args.threads, extra=load_extra(d), s1_frac=args.s1_frac,
                        chunk_s1=args.chunk_s1)
    if args.dataset == "train":
        meta = pl.read_parquet(os.path.join(d, "s1_meta.parquet"))
        if args.s1_frac < 1.0:
            meta = meta.filter((pl.col("s1_idx").hash(5) % 10_000) < int(args.s1_frac * 10_000))
        gt = pl.read_parquet(os.path.join(d, "gt_pairs.parquet")).join(meta.select("s1_idx"), on="s1_idx")
        cand = pl.read_parquet(out, columns=["s1_idx", "cand_idx", "blk_rank"])
        log(f"blocking report: {blocking_report(cand, gt, meta)}")


if __name__ == "__main__":
    main()
