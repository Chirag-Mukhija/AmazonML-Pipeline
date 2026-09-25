# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary

We use a three-stage pipeline: rare-key **blocking**, a **LightGBM pairwise matcher**, and an **F0.5-aware decision layer**. Blocking turns names and addresses into IDF-weighted sparse key vectors and runs a multithreaded top-K sparse product per country, so no full pairwise comparison is needed. The matcher uses about 70 country-agnostic features. The key additions are (a) a transliteration dictionary for Indic scripts learned from the training pairs alone, (b) "competition" features built on the fact that every Source-2/3 record matches at most one Source-1 entity, and (c) a decision rule that enforces one-to-one assignment and is tuned directly for macro F0.5, including singleton handling.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from EDA on the 2.2M / 5.0M / 5.3M training records and their 7.64M true pairs:

- **Each S2/S3 record matches at most one S1 entity.** Of 7.64M matched ids, all are unique, and about 26% of S2/S3 records are unmatched distractors. We use this as a hard one-to-one constraint and as a family of features.
- **Country always agrees** between matched records, so every stage is scoped per country. Country is treated as an open label set: the test set adds France (15% of test S1), which does not appear in training.
- **Cardinality:** 5.6% of S1 entities are singletons. Most others have 2–6 matches, up to 5 from S2 and 6 from S3, because the sources contain duplicates within themselves.
- **Name noise:** legal-form swaps (Pvt/Private, Ltd/Limited, Inc/Incorporated), injected filler words (Center, Partners, Services, Shri, Dr), duplicated tokens ("Sequoia Sequoia"), token reordering, character typos ("Haelhtacre"), accent noise ("Sérvice"), website forms ("desertcapitalkkr.com", "#grandmitsui"), `d/b/a` trade names, and names made of unrelated gibberish.
- **Scripts:** in India, 23% of matched S2 names and 13% of matched S3 names are in Devanagari, Tamil, Telugu or Gurmukhi. Source 1 names are always Latin. These are phonetic transliterations of English names ("विजन प्रोडक्ट्स प्राइवेट लिमिटेड" = "Vision Products Private Limited").
- **Address noise:** component reordering (city first), `<NULL>`/`NULL` tokens, full state names vs codes vs native script (Kerala/Keralam/KL/केरल), house-number corruption (00855 vs 855, 1630 vs 630), added PMB / PO BOX / PLOT / HN prefixes, city substitution (township vs city), and 3–4% empty addresses. No PIN codes appear in India addresses, and only about 11% of US addresses carry a ZIP.

### 2.2 Solution Strategy

**Approach Type:** Blocking + gradient-boosted pairwise classifier + constrained decision layer (hybrid, graph-aware)  
**Core Innovation:** (1) a transliteration dictionary learned only from labelled training pairs; (2) competition features plus one-to-one assignment derived from the "each record belongs to at most one entity" structure; (3) an expected-F0.5 / threshold decision rule tuned directly on the macro metric.

---

## 3. Candidate Generation (Blocking)

Each record emits keys on two channels:

| channel | keys |
|---|---|
| name | normalized tokens; consonant skeleton per token (typos and transliteration); compact name without spaces (catches websites); first 6 characters of the compact name |
| address | house-number + street-word; compact alphanumeric ids (`B-11/2` and `B-1-1/2` both become `b112`); numbers with 3+ digits; address token bigrams |

- **Weighting and pruning:** a key is weighted by IDF over the S2+S3 pool. Keys touching more than 0.2% of the pool are dropped because they carry little information and dominate cost.
- **Search:** per country and per channel, `Q (S1 x keys) · P^T (keys x pool)` is computed with `sparse_dot_topn`, keeping the top 40 per S1. The union of both channels is scored on both channels and capped at the 60 best by combined score. That capped set is what the matcher scores, and it is written to `candidate_pairs.tsv`.
- **Optional embedding channel (GPU):** multilingual-e5 embeddings with FAISS kNN top-30. These hits always survive the cap.
- **Candidate pairs generated:** about 66 per Source-1 entity (about 146M for full train and about 115M for test, projected from full-density runs). The v1 two-channel design averaged 59 per entity.
- **How true matches were kept:** two independent channels (name and address), so a candidate with a gibberish trade name is still reachable through its address, and one with an empty address through its name. Skeleton and compact keys absorb typos and website forms. The learned transliteration dictionary makes Indic-script names share tokens with their Latin counterparts. Recall and the F0.5 ceiling are measured on training data after every change (`blocking.py` report).

---

## 4. Matching Model

**Features used (about 70, none encode the country):**
- Name features: rapidfuzz ratio / partial / token-sort / token-set / WRatio on the core name; ratio, partial and Jaro-Winkler on the compact name; skeleton ratio; token and skeleton Jaccard and containment in both directions; first-token equality; compact-name containment; legal-form agreement; token and character lengths.
- Address features: ratio / token-set / partial / token-sort; token Jaccard and containment; number-set overlap and containment; house-number equality and Levenshtein distance; street-key equality; alphanumeric-id overlap; empty-address flags (address similarities set to null, not 0, when an address is missing).
- Blocking features: IDF-weighted name score, address score, combined score, and rank.
- Context features: per S1, the rank of and gap to the best candidate on name, address and blocking score, plus the number of strong candidates. Per candidate, the number of S1 entities competing for it, this S1's rank among them, and the gap to the candidate's best S1.
- Other: candidate script flag (non-Latin), website flag, source (S2/S3). Optional `emb_cos` when embeddings are enabled.

**Model type:** LightGBM binary classifier (127 leaves, lr 0.05, early stopping on the validation split).  
**Threshold selection method:** Grid search over (a) a probability threshold and (b) an expected-F0.5 rule, which picks the top-k maximizing `1.25·Σp_i / (0.25·Σp + k)` and predicts empty when `Π(1−p_i)` is higher. Both are applied after one-to-one assignment. The rule is chosen by macro F0.5 on a hash-held-out 20% of S1 entities, with every singleton and every no-candidate entity counted.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, validation):** 0.9887 on the 5% train mini-world (pair precision 0.996, pair recall 0.976, singleton accuracy 0.984). [TBD: full-density number from the VM run]
- **Blocking at full density** (5% of train S1 blocked against the complete 10.3M-record pool, which gives exact per-entity results):

  | version | pair recall | F0.5 ceiling | candidates / S1 |
  |---|---|---|---|
  | v1: 2 channels, re-ranked cap | 0.943 | 0.979 | 59 |
  | v2: + OCR/legal/ID normalization, 3 un-re-ranked channels | **0.966** | **0.989** | 66 |

- **Common false positives (wrong merges):** same name on the same street with a different house number (22 vs 31 Vine St; 644 vs 648 Fox Run Trail), which the generator uses as hard negatives. Also identical names where the candidate has an empty address and two S1 entities compete for it. Mitigations: house-number suffix/prefix/difference features, and abstaining when a candidate's two best S1 scores are within `amb`.
- **Common false negatives (missed matches):** names replaced by an unrelated trade name ("Evoyuma") combined with a weak address; initials-style websites (`nbrothers.com`); candidates with an empty address where the model cannot verify the location (p between 0.25 and 0.7). OCR noise (`5TAY`, `YOUN6`, `F0undation`), dotted legal forms (`L.L.C.`) and phone/ID suffixes were fixed in v2 normalization.

---

## 6. Conclusion

[TBD]

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/src/`: `normalize.py`, `prepare.py`, `blocking.py`, `embed.py`, `features.py`, `train.py`, `decide.py`, `analyze.py`, `scoring.py`, `run.py` (entry point), `package.py`. To reproduce:

```bash
cd code/business_entity_resolution && pip install -r requirements.txt
python src/run.py all --data-dir ../../dataset --work-dir ../../work --model-dir ../../models --output-dir ../../output
```

### B. Additional Results

- Top LightGBM features by gain on the mini-world: `cx_cand_rank`, `cx_cand_gap`, `atok_cont2`, `ad_tset`, `blk_rank_both`, `nm_partial`, `atok_jac`, `nm_full_tset`, `anum_jac`, `nc_jw`. The two competition features lead, which confirms the value of the one-S1-per-record structure.
- Per-country validation F0.5 on the v1 mini-world: US 0.9876, India 0.9860.
- [TBD: full-density per-country F0.5 and recall@K from the VM run]

---

**Compliance:** no external data, APIs or geocoding. The transliteration dictionary is learned from the training pairs, and `unidecode` is a static bundled table. Models: LightGBM (MIT); optional `intfloat/multilingual-e5-base` (MIT, 278M parameters).
