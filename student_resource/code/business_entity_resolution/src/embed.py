"""Optional (GPU box): dense multilingual embeddings as an extra blocking
channel and an extra matcher feature.

Model default: intfloat/multilingual-e5-base (MIT license, 278M params) --
covers Hindi/Tamil/Telugu/Punjabi scripts and French. Swap for
BAAI/bge-m3 (MIT, 568M) on a big GPU. Both are far under the 8B limit.

  python src/embed.py encode --work-dir work --dataset train [--model ...]
  python src/embed.py block  --work-dir work --dataset train --k 30

`encode` writes <work>/<dataset>/emb.npy (float16, row i = record idx i).
`block` writes <work>/<dataset>/emb_candidates.parquet (s1_idx, cand_idx, emb_cos);
blocking.py unions it into candidates.parquet when present, and features.py
adds `emb_cos` for every candidate pair when emb.npy exists.
"""
import argparse
import os
import time

import numpy as np
import polars as pl

from io_utils import log

DEFAULT_MODEL = "intfloat/multilingual-e5-base"


def record_texts(records: pl.DataFrame) -> list:
    return records.select(
        (pl.lit("query: ") + pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null(""))
    ).to_series().to_list()


def encode(work_dir: str, dataset: str, model_name: str, batch_size: int, device: str = None):
    from sentence_transformers import SentenceTransformer
    import torch

    d = os.path.join(work_dir, dataset)
    records = pl.read_parquet(os.path.join(d, "records.parquet"),
                              columns=["idx", "business_name", "business_address"]).sort("idx")
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = SentenceTransformer(model_name, device=device)
    if device == "cuda":
        model.half()
    texts = record_texts(records)
    t = time.time()
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True,
                       convert_to_numpy=True).astype(np.float16)
    np.save(os.path.join(d, "emb.npy"), emb)
    log(f"encoded {len(texts)} records with {model_name} on {device} in {time.time() - t:.0f}s -> {emb.shape}")


def knn_block(work_dir: str, dataset: str, k: int, use_gpu: bool = True):
    import faiss

    d = os.path.join(work_dir, dataset)
    emb = np.load(os.path.join(d, "emb.npy"), mmap_mode="r")
    records = pl.read_parquet(os.path.join(d, "records.parquet"), columns=["idx", "src", "country"])
    out = []
    for country in records["country"].unique().sort().to_list():
        rc = records.filter(pl.col("country") == country)
        q_idx = rc.filter(pl.col("src") == 1)["idx"].to_numpy()
        p_idx = rc.filter(pl.col("src") != 1)["idx"].to_numpy()
        if len(q_idx) == 0 or len(p_idx) == 0:
            continue
        xb = np.ascontiguousarray(emb[p_idx].astype(np.float32))
        index = faiss.IndexFlatIP(xb.shape[1])
        if use_gpu and faiss.get_num_gpus() > 0:
            index = faiss.index_cpu_to_all_gpus(index)
        index.add(xb)
        for s in range(0, len(q_idx), 200_000):
            qi = q_idx[s:s + 200_000]
            sims, nn = index.search(np.ascontiguousarray(emb[qi].astype(np.float32)), k)
            out.append(pl.DataFrame({
                "s1_idx": np.repeat(qi, k).astype(np.int32),
                "cand_idx": p_idx[nn.ravel()].astype(np.int32),
                "emb_cos": sims.ravel().astype(np.float32),
            }))
        log(f"  emb knn country={country!r}: {len(q_idx)} queries x {len(p_idx)} pool")
    pl.concat(out).write_parquet(os.path.join(d, "emb_candidates.parquet"))


def pair_cosine(emb: np.ndarray, a: np.ndarray, b: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    out = np.empty(len(a), dtype=np.float32)
    for s in range(0, len(a), chunk):
        x = emb[a[s:s + chunk]].astype(np.float32)
        y = emb[b[s:s + chunk]].astype(np.float32)
        out[s:s + chunk] = (x * y).sum(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["encode", "block"])
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--dataset", choices=["train", "test"], required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default=None)
    ap.add_argument("--k", type=int, default=30)
    args = ap.parse_args()
    if args.cmd == "encode":
        encode(args.work_dir, args.dataset, args.model, args.batch_size, args.device)
    else:
        knn_block(args.work_dir, args.dataset, args.k)


if __name__ == "__main__":
    main()
