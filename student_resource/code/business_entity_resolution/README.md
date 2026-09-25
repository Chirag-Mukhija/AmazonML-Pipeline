# Business Entity Resolution — Pipeline

Blocking → pairwise features → LightGBM matcher → F0.5-aware decision rule.
Methodology write-up: `Documentation_template.md` at the zip root.

```
raw TSVs ──prepare──► records.parquet (normalized) ──blocking──► candidates.parquet
         ──features──► features/part-*.parquet ──train──► model.txt ──decide──► matching_results.tsv
```

## Setup

```bash
cd code/business_entity_resolution
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# macOS only: LightGBM needs OpenMP ->  brew install libomp
```

## Reproduce end to end

Run from `code/business_entity_resolution/`. `--data-dir` is the folder holding
`train/` and `test/` (i.e. `student_resource/dataset`).

```bash
python src/run.py all \
    --data-dir ../../dataset \
    --work-dir ../../work \
    --model-dir ../../models \
    --output-dir ../../output
```

`train` = prepare → block → featurize → train LightGBM → tune the decision rule
on the held-out 20% of Source-1 entities (prints the validation macro F0.5 and
writes `models/val_report.json`).
`predict` = the same stages on `dataset/test`, then writes
`output/matching_results.tsv` + `output/candidate_pairs.tsv` and runs
`utils/validate_submission.py` on them.

Stages cache to `--work-dir`; add `--skip-existing` to reuse them while
iterating on a later stage (delete a stage's output to force a rerun).

### Laptop-sized runs

| flag | effect |
|---|---|
| `--sample-frac 0.05` | train on a 5% "mini-world" (sampled S1 + all their matches + 5% of distractors) — ~2 min end to end |
| `--feature-s1-frac 0.2` | block against the full pool, but only featurize/train on 20% of train S1 (memory/disk) |
| `--threads 4` | leave cores free |

Full train on the GPU VM: no sampling flags. Blocking and features both stream
S1 in chunks (`src/blocking.py --chunk-s1`, `run.py --chunk-s1`), so their RAM
stays bounded. On a 16 GB laptop, full-scale blocking fits with 150k-S1 chunks,
but it takes about an hour and stalls whenever macOS sleeps, so plug in and run it
under `caffeinate -i`. LightGBM loads all fit rows at once: use `--neg-frac 0.3`
if RAM is short (probabilities are re-calibrated automatically). Disk: about
50 MB of features per million candidate pairs.

## Stages (each also runnable alone, see `--help`)

| file | what it does |
|---|---|
| `src/normalize.py` | transliteration (learned dictionary + unidecode), legal-form stripping, address canonicalization (EN/FR/IN abbreviations), house numbers, alnum ids |
| `src/prepare.py` | loads TSVs, fit/val split by hashed S1 id, learns translit dictionary from **fit** pairs only, normalizes, caches parquet |
| `src/blocking.py` | rare-key blocking: IDF-weighted name/address keys → sparse top-K product per country (sparse_dot_topn); reports pair recall + F0.5 ceiling |
| `src/embed.py` | *optional, GPU*: multilingual-e5 embeddings → FAISS kNN extra candidate channel + `emb_cos` feature |
| `src/features.py` | ~70 features: rapidfuzz similarities (vectorized `cpdist`), token/number/id overlaps, competition context features |
| `src/train.py` | LightGBM with early stopping on the val split; saves model, feature list, importances |
| `src/decide.py` | one-to-one assignment + threshold / expected-F0.5 rule, grid-tuned on val |
| `src/analyze.py` | per-country F0.5, dumps top false positives / false negatives to TSV |
| `src/scoring.py` | standalone macro-F0.5 scorer for any two TSVs |
| `src/package.py` | builds `<team>_submission.zip` in the required layout |

## Using the embedding channel (GPU box)

```bash
pip install torch sentence-transformers faiss-gpu-cu12
python src/embed.py encode --work-dir ../../work --dataset train
python src/embed.py block  --work-dir ../../work --dataset train --k 30
python src/embed.py encode --work-dir ../../work --dataset test
python src/embed.py block  --work-dir ../../work --dataset test --k 30
# then re-run blocking + features + train (delete candidates.parquet / features/ first)
```

## Validate and package

```bash
cd ../..    # student_resource/
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
python3 code/business_entity_resolution/src/package.py --team <team_name>
```

## Fast experiment loops

```bash
# exact blocking recall for 5% of train S1 against the FULL pool (~3 min on a laptop)
python src/blocking.py --work-dir ../../work --dataset train --s1-frac 0.05 [--k-both 60 ...]
# per-country F0.5 + top false positives / false negatives as TSVs
python src/analyze.py --work-dir ../../work --model-dir ../../models
# regression tests (normalization cases, scorer vs problem-statement example, decision rules)
python tests/test_pipeline.py
```

## Known gaps / next steps (v1 → v2)

Blocking (recall ceiling)
- Full-density pair recall is ~96.6% (F0.5 ceiling ~0.989). Look at the misses with an `--s1-frac` run: they are mostly gibberish trade names plus weak or empty addresses, and initials-style websites (`nbrothers.com`).
- Run `embed.py` on the GPU (multilingual-e5 or bge-m3) as an extra channel, then compare recall with the embedding channel on and off.
- Tune `--k-*` and `--max-df-frac` on the VM, where more cores make a larger K cheap.

Matcher
- Train on all 2.2M train S1 on the VM (no `--feature-s1-frac`), then tune LightGBM parameters.
- Stage-2 stacking: feed each candidate's best first-stage p, the S1's p-rank and similar scores into a second model. This is competition computed on model scores rather than blocking scores.
- Transitivity features: S2 and S3 candidates of the same S1 that strongly match each other.
- `emb_cos` feature; a cross-encoder reranker fine-tuned on training pairs (MIT/Apache, ≤8B).
- State/region equivalence learned from training pairs (Kerala/KL/केरल). France cannot be learned this way.

France (unseen in train)
- Spot-check test predictions for France and compare the match rate per country on test with train (about 94% of S1 have matches).
- Normalization already covers French legal forms (SARL/SAS/SASU/SA/SCI/EURL) and street types (R/Rue, BD, AV, ALL, IMP, CHEM, RTE).

## Rules compliance

- No external data/APIs. The transliteration dictionary is learned from the
  training pairs; `unidecode` is a static, bundled character table.
- Models: LightGBM (MIT). Optional embedding model `intfloat/multilingual-e5-base`
  (MIT, 278M params) — well under the 8B limit.
- Country is treated as an open label set: blocking iterates over whatever
  countries appear; no feature encodes the country itself (France is unseen in train).
