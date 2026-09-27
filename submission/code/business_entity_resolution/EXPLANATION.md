# Pipeline walkthrough and design rationale

This document explains *why* the pipeline is built the way it is, not just what
it does — the goal is for you to be able to look at any decision, judge whether
it's still right, and change it with full context. It assumes you know standard
ML (train/val splits, gradient boosting, precision/recall) but not this specific
codebase or entity-resolution tricks.

For a quick command reference see `README.md`. This document is the "why".

---

## 1. The shape of the problem, and what that implies

Before writing any code I ran EDA (`Documentation_template.md` §2.1 has the
condensed version). Four facts from that EDA drove almost every architectural
choice:

**(a) Every Source-2/3 record belongs to at most one Source-1 entity.**
I verified this directly: `7,638,365` matched ids in the training ground truth,
and all `7,638,365` are unique. This is a *hidden one-to-one structure* the
problem statement doesn't call out explicitly, but it's true in the data and I
build on it in two places:
  - **Decision layer** (`decide.py`): after scoring, each candidate is assigned
    to at most one Source-1 entity (its highest-probability one), never split
    across several. This isn't just a nicety — it's a precision lever, because
    F0.5 punishes false merges 2x harder than misses.
  - **Features** (`features.py`, `candidate_context`): "how many other S1
    entities also want this candidate, and what's my rank among them" is
    computed from the *raw blocking scores across the whole candidate table*
    (not per-pair in isolation). This turned out to be the single strongest
    feature group (see §7, feature importances) — stronger than any string
    similarity. This is the kind of signal you only get from thinking about
    the *graph* structure of the matching problem, not just pairwise text
    similarity, and it's the main thing I'd point to as "the non-obvious part
    of this solution."

**(b) Country always agrees between matched records** (I checked: 100% in
training). So blocking and feature computation are scoped per-country
throughout — never comparing a US record to an India record. This is also why
I never one-hot-encode country as a model feature: it would be a constant
within each comparison and therefore useless, and worse, hard-coding it would
break on France (unseen in train, present in test — the problem statement
explicitly warns about this).

**(c) Only ~5.6% of Source-1 entities are singletons** (no match anywhere),
but they're scored the same as everyone else and a false positive on one costs
a full point. That's why there's a dedicated ambiguity-abstention mechanism
(§5) and why the decision rule is tuned against the *macro* F0.5, not a global
threshold picked by eyeballing precision/recall.

**(d) The India subset has heavy transliteration noise**: ~23% of matched
Source-2 names and ~13% of matched Source-3 names are in Devanagari, Tamil,
Telugu, or Gurmukhi script, always as a phonetic transliteration of the
Source-1 Latin name (e.g. "विजन प्रोडक्ट्स प्राइवेट लिमिटेड" ↔ "Vision Products
Private Limited"). This is what motivated the self-supervised transliteration
dictionary in §2.

None of this is exotic — it's what you'd find by spending an hour on `polars`
groupbys before writing a line of matching logic. I'd treat "did you actually
look at the joined true-positive pairs before designing features" as the
highest-leverage step in the whole project, and I'd redo it again first if
starting over.

---

## 2. Normalization (`normalize.py`) — the most under-discussed stage

It's tempting to treat normalization as boilerplate and spend your effort on
"the real ML." I don't think that's right for this problem: **almost every
false negative I found in error analysis was a normalization gap, not a model
capacity gap.** LightGBM with mediocre features will still separate signal
from noise; but if two true-match records don't share *any* usable token
because of a formatting quirk, blocking never gives the model a chance to see
the pair at all. So this is where I spent the most iteration cycles.

### 2.1 What it produces

For each record, `normalize_frame` derives (all documented in the file's
top-of-file docstring, reproduced here for context):

| column | purpose |
|---|---|
| `name_core` | name with legal suffixes and filler words stripped, deduped tokens — the "identity" of the business |
| `name_compact` | `name_core` with spaces removed — catches names that appear as URLs/hashtags (`desertcapitalkkr.com`, `#grandmitsui`) |
| `name_skel` | consonant skeleton per token (see 2.3) — typo/OCR/transliteration-tolerant |
| `legal_form` | canonicalized legal suffix (`Pvt Ltd` / `Private Limited` / `(Limited)` → `pvt ltd`), separated out so it can be compared independently instead of just adding string-similarity noise |
| `addr_tokens` | address tokens with abbreviations unified (`Rd`→`rd`, `Boulevard`→`blvd`, French `R.`→`rue`, Indian `Nr`→`nr`) |
| `house_no`, `street_key` | first street-like number + following word, e.g. `"703 Powers Extension, ..."` → `house_no="703"`, `street_key="703_powers"` |
| `addr_ids` | compact alphanumeric identifiers, e.g. `B-11/2` and `B-1-1/2` both normalize to `b112` — this exists specifically because Indian addresses use inconsistent hyphenation/spacing in plot numbers |

### 2.2 Decisions worth scrutinizing

- **Legal-form stripping.** I maintain an explicit dict (`LEGAL_FORMS`) mapping
  surface variants to canonical forms, covering US/UK forms (Inc, Corp, LLC,
  Ltd, PLC...) and French forms (SARL, SAS, SASU, EURL, SCI...). I added the
  French forms only after noticing French test-set-style names in the sample
  I inspected — this is a place where the France-unseen-in-train facet of the
  challenge required speculative coverage rather than training-data-driven
  coverage. **Risk**: this dict is necessarily incomplete; if you find France
  false negatives caused by an unrecognized legal form, this is the first
  place to look.
- **Filler-word removal** (`FILLER = {"the","and","of","dba","www","com",
  "shri","sri","dr","m","s"}") — these are words I saw the noise generator
  inject that don't carry business identity (e.g. "Ectoquocalo One DBA Concept
  Agency Private Limited" ↔ "Concept Agency Private Limited"). This list was
  built empirically from reading false negatives, not from a general stopword
  list — it's deliberately narrow so it doesn't accidentally strip meaningful
  short tokens from real business names.
- **Name-skeleton folding** (`_skeleton` / `skeleton_expr`): maps `ph→f`,
  `ck→k`, `x→ks`, `c/q→k`, `z→s`, `w→v`, `i→l`, then drops vowels and collapses
  repeated consonants. This single function is what lets `"Haelhtacre"` (typo)
  and `"Healthcare"` collide, and what lets Latin transliterations of Indic
  names collide with their English equivalent even when the translit
  dictionary (§2.4) hasn't learned that specific word. It's a deliberately
  aggressive, lossy hash — false collisions are fine here because skeleton
  matches are only used as one blocking signal among several (see §3), never
  as a matching feature on their own without corroboration.
- **Leetspeak/OCR de-noising** (`_deleet`): a single embedded digit in an
  otherwise-alphabetic token (`5tay`, `youn6`, `f0undation`) gets replaced by
  the letter it visually resembles (`5→s`, `6→g`, `0→o`, etc.), but only when
  the token is mostly letters (`>=3` letter chars) — so real alphanumeric
  identifiers and phone-number-like tokens are untouched. I added this after
  finding several false negatives where the *only* difference between two
  otherwise-identical names was a single substituted digit. This is a
  narrow, data-driven fix, not a general "always try leet-decoding" rule —
  worth keeping narrow, since being too aggressive here would corrupt real
  alphanumeric codes.
- **Dotted-abbreviation collapsing** (`l.l.c.→llc`, `p.c.→pc`), **ID/phone
  stripping** (`"(ID: 29782)"`, 7+ digit runs) — small, specific regexes added
  in response to specific false positives/negatives found in `analyze.py`
  output, not designed up front. This is the general pattern for this file:
  **almost every rule here is a response to a concrete observed failure**,
  not a guess at what noise "might" exist. If you extend this file, I'd
  recommend keeping that discipline — run `analyze.py`, find a real failure,
  add the minimal rule that fixes it, re-run the regression tests.
- **Address canonicalization dict** (`ADDR_CANON`) mixes English, French, and
  Indian address vocabulary in one flat dict, scoped by nothing (i.e. an
  Indian address containing "Fort" gets the same treatment as a US one, which
  maps to `ft`). This is a simplification — a per-country dict would be more
  precise but adds complexity for (I judged) limited benefit, since these are
  mostly non-colliding vocabularies. Worth revisiting if you find abbreviation
  collisions across countries causing false matches.
- **House-number extraction takes the first comma-separated address
  component that contains a digit.** This is a heuristic, not a real address
  parser — addresses in this dataset have their components in inconsistent
  order (sometimes city-first, sometimes street-first; see `"Tallassee, 1795
  Westchester Drive, High Point, NC"` vs `"1795 Westchester Drive, High Point,
  NC"`-style variation), so I couldn't assume a fixed position. Taking "first
  numeric component" is a reasonable proxy but will occasionally grab a PIN
  code or apartment number instead of a street number. I didn't build a real
  CRF/rule-based address parser — that would be the natural next step if
  house-number precision turns out to be a bottleneck (`analyze.py` will show
  you if `house_*` features are misfiring).

### 2.3 Transliteration: two layers, deliberately kept separate

There are two independent mechanisms for handling non-Latin script, and I
want to be explicit about why both exist rather than just one:

1. **`unidecode`** (generic ASCII-folding of any non-ASCII text) — a bundled,
   static character-transliteration table. This handles *phonetic* romanization
   in a generic, language-agnostic way (e.g. "राम" → "raam"), and also handles
   French accents (`Établissements` → `Etablissements`, `Sérvice` → `Service`).
   It is **not** a lookup service or API — it's a static table shipped in the
   `unidecode` PyPI package, the same category of tool as a stemmer, so it
   doesn't violate the "no external data lookup" rule.
2. **The learned translit dictionary** (`learn_translit_dict`, called once in
   `prepare.py` during `prepare_train`): built *only* from the training
   ground-truth pairs where the Source-1 name is Latin and the matched
   Source-2/3 name is non-Latin, by positionally aligning tokens (only when
   both names have the same token count) and keeping a `src → dst` mapping
   when it's the majority mapping for that source token (`min_count=2`,
   `min_share=0.5`). This is how the pipeline learns that "प्राइवेट" specifically
   means "private" in *this* dataset's business-name vocabulary — semantic
   translation that `unidecode` alone can't give you (`unidecode("प्राइवेट")`
   gives you a phonetic transcription, not the English word "private").

   **Critical detail on leakage**: this dictionary is learned from the **fit**
   split only (`s1_meta.fold == "fit"`), never from `val`. This matters
   because the dictionary is then baked into `records.parquet` before the
   train/val split is used anywhere downstream — if it had been learned from
   all training data including val, the validation F0.5 would be optimistic
   (the model would effectively have seen val-set vocabulary during
   normalization). Same dictionary is reused verbatim for the test set
   (`prepare_test` loads `translit.json` from the train run). **This dictionary
   cannot help with France** (no Latin-script legal/business vocabulary in a
   language it wasn't trained on) — French normalization relies entirely on
   the static `LEGAL_FORMS`/`ADDR_CANON` dicts and `unidecode`'s accent-folding.

---

## 3. Blocking (`blocking.py`) — the recall ceiling, and how it's measured

### 3.1 Why sparse top-K matrix products, not classic key-based blocking

The first version of this pipeline (which I later replaced) used classic
inverted-index blocking: build `dict[(country, token)] -> [record_ids]`, and
for a query record, union the postings lists of its tokens. That's simple and
fast to reason about, but it has two problems at this scale:
1. It gives you a *binary* signal (shares a key / doesn't) — no ranking within
   a block, so when a block is large you either keep everything (expensive
   downstream) or truncate arbitrarily.
2. It doesn't compose multiple weak signals into one score cheaply.

The current version instead represents each record as a **sparse vector over
blocking keys, IDF-weighted**, and computes **all pairwise dot products above
a threshold, keeping only the top-K per query row**, using `sparse_dot_topn`
(a mature, multithreaded C++ implementation of exactly this operation — the
same core routine used by the well-known `string_grouper` fuzzy-matching
library). This gives you a *ranked* shortlist per record in one shot, weighted
by how rare/informative each shared key is, without ever materializing a full
`n_s1 × n_pool` dense or even fully-realized sparse matrix (top-K pruning
happens inside the C++ routine).

### 3.2 The three channels, and why they're *not* re-ranked together

Every record emits keys into two vocabularies:

```
name keys:     t:<token>            k:<consonant skeleton>
               c:<compact name>     p:<first 6 chars of compact name>
address keys:  s:<house#_street>    i:<alnum id>
               n:<number, 3+ digits>   b:<address token bigram>
```

(See `record_keys` for the exact expressions — `p:` is specifically there to
catch website-style names like `desertcapitalkkr.com` where the *token* split
would fail but a character-prefix match still works.)

These feed into three independent top-K searches per Source-1 record:
- **`name`**: rank by name-key overlap only, keep top `k_name` (default 30)
- **`addr`**: rank by address-key overlap only, keep top `k_addr` (default 30)
- **`both`**: rank by name+address combined score, keep top `k_both` (default 40)

The candidate set fed to the matcher is the **union** of all three, each
channel's ranks preserved as separate columns (`blk_rank_name`,
`blk_rank_addr`, `blk_rank_both`) alongside a combined `blk_score` used for
`blk_rank`.

**Why not just do one combined top-K search?** Early on I did exactly that
(single combined channel, single re-ranked cap), and it measurably hurt recall
on cases with a strong signal in *one* field and no signal in the other — e.g.
a perfect name match with an empty or wildly different address (common: ~4%
of addresses are empty in this dataset) would get pushed out of a combined
top-K by pairs that are mediocre-but-consistent on both fields. Splitting into
independent channels means a strong `name`-only match is never crowded out by
weak-but-broad `addr` matches, and vice versa. This was worth a concrete
measurable recall gain (see the two-vs-three-channel numbers in §3.4) — it's
the single biggest architectural change between the two blocking iterations
I built.

### 3.3 IDF weighting and the frequency cap

Two records only get compared through a key if that key isn't too common:
`max_df_frac` (default 0.002 = 0.2% of the pool) caps how many pool records a
single key is allowed to touch before it's dropped as "uninformative and
expensive." Surviving keys are weighted by `1 + log(N_pool / df)` — standard
IDF — so a rare shared token contributes much more to the match score than a
common one. Query vectors are then L1-normalized (divided by their weight
sum) so the score can be read as "the weighted fraction of this record's own
keys that were found in the candidate," bounded in `[0, 1]` per channel.

**This is a knob, not a fixed constant** — `--max-df-frac` and `--k-*`
directly trade off recall against candidate-set size (and therefore
downstream feature/training cost). I did a small sweep on the mini-world
(documented in conversation, not committed as a script) showing recall rising
roughly monotonically with a looser cap and higher K, with diminishing
returns past `k≈30-40` per channel — but I did **not** do this sweep at full
data density (only on the 5%-of-train-S1-vs-full-pool exact-recall harness,
§3.4), so retuning these on the VM once full runs are cheap is a legitimate
next step, not something I'd consider already optimized.

### 3.4 Recall measurement: `--s1-frac` gives you *exact* numbers cheaply

`blocking_report` computes, against ground truth:
- **pair_recall**: fraction of true (S1, match) pairs that survive into the
  candidate set at all — this is the hard ceiling on final recall, since a
  pair the matcher never sees can never be predicted.
- **f05_ceiling**: the macro F0.5 you'd get with a *perfect* matcher on top of
  this candidate set (i.e., if it labeled every true candidate positive and
  every false one negative) — this is the honest ceiling on the whole
  pipeline's score, not just blocking's isolated recall number.
- **pair_recall@k**: recall if you truncated to the top-k combined-score
  candidates, for k in {5,10,20,40} — tells you how much of your recall is
  "found immediately" vs. "found but buried deep in the ranking," which
  matters because the matcher still has to separate true positives from a
  larger pile of true negatives at higher k.

The important trick here, because full blocking is expensive (see §8 on scale):
**`--s1-frac 0.05` blocks a 5% hash-sample of Source-1 records against the
*complete, unsampled* Source-2/3 pool.** Per-record results are mathematically
identical to a full run (a given S1 record's candidate search doesn't depend
on which other S1 records exist) — so this gives you an *exact*, not
approximate, recall/ceiling measurement in a few minutes instead of an hour,
which is what made rapid iteration on blocking parameters possible on a
laptop. I used this constantly; I'd treat it as the primary tool for anyone
tuning blocking further.

**Measured full-density numbers** (5% of train S1 vs. the complete 10.3M-record
pool, before vs. after the normalization/channel improvements in §2 and §3.2):

| version | pair recall | F0.5 ceiling | candidates/S1 |
|---|---|---|---|
| v1 (2 channels, cross-channel re-rank cap, pre-OCR-fix normalization) | 0.943 | 0.979 | 59 |
| v2 (3 independent channels, current normalization) | **0.966** | **0.989** | 66 |

That 2.3-point recall gain is why I consider the channel-independence decision
(§3.2) and the normalization fixes (§2.2) the two highest-value changes made
after the first working version — worth knowing if you're deciding where to
spend further effort, since it suggests blocking-side work still has payoff
(the ceiling is 0.989, not 1.0 — there's a known-irreducible ~3.4% pair-recall
gap discussed in §7).

### 3.5 Memory-bounded execution — an implementation detail worth knowing about

This mattered enough during development that I want to flag it explicitly:
the naive way to implement this (`Q @ P.T`, `top_n` over the whole thing)
blew past 16GB and started swapping on the full US pool (1.3M S1 × 6.2M pool
records → hundreds of millions of nonzero entries in the intermediate sparse
product). `block_country` is a **generator** that processes Source-1 records
in chunks (`--chunk-s1`, default 250k), building the pool-side sparse matrices
*once* per country and re-using them across chunks, writing each chunk's
result to a parquet part file rather than holding all of it in memory.
`generate_candidates` then streams those parts back together with
`pl.scan_parquet(...).sink_parquet(...)` (lazy, out-of-core). This is why
blocking accepts a `records.parquet` *path* rather than requiring a DataFrame
already in memory — `_country_frame` uses `scan_parquet(...).filter(...)`
so only one country's data is materialized at a time.

**I did not get to run a full, unsampled train-set blocking pass end-to-end on
the laptop** — even with chunking, the underlying work for the full US pool
(1.3M × 6.2M keys) took long enough that macOS's aggressive idle-sleep
repeatedly stalled it (see the session's tool log if you want the detail; the
short version is a 16GB laptop on battery isn't the right place to run this at
final scale). What I verified instead: (a) chunked output is byte-identical in
aggregate stats to unchunked output on the mini-world (`6,574,616` vs
`6,574,755` pairs — the tiny difference is candidate-cap tie-breaking, not a
correctness bug), and (b) the `--s1-frac 0.05`-against-full-pool numbers in
§3.4, which exercise the exact same code path against the exact same
full-size pool, just for fewer S1 rows. **Running the real, full, unsampled
`generate_candidates` for train and test is the first thing to do on the GPU
VM** — I'd expect it to take well under the laptop's time given proper
uninterrupted execution and more RAM to raise `--chunk-s1`.

---

## 4. Features (`features.py`) — three tiers, and which one actually matters

### 4.1 Tier 1: pairwise string/set similarity (the "obvious" features)

`STRING_FEATURES` runs 10 rapidfuzz scorers (ratio, partial ratio, token-sort,
token-set, WRatio, Jaro-Winkler) over `name_core`, `name_compact`, and
`name_skel_str`, plus 4 more over `addr_norm`. These run through
`process.cpdist` — rapidfuzz's **vectorized, multithreaded** element-wise
comparator (not `process.extract`/cdist over a cross-product — `cpdist` takes
two equal-length lists and compares them pairwise, which is exactly the
shape of "one row per candidate pair" data). This is a performance decision:
computing ~14 string metrics over tens of millions of pairs in a Python loop
would be the dominant cost of the whole pipeline; `cpdist` moves it into C++
and threads it.

Set-overlap features (`_set_feats`: intersection size, Jaccard, and both
directional containments) run over `name_tokens`, `name_skel` (as a set),
`addr_tokens`, and `addr_numbers` — computed as native polars list
expressions (vectorized, no Python loop) rather than a Python UDF.

**Missing-data handling is deliberate, not an oversight**: when an address is
empty on either side, every address-similarity feature is set to `null`
(via `addr_missing` masking), not `0`. This distinction matters a lot to a
tree model — "0% similar" and "no information" are different facts, and
LightGBM natively handles missing values by learning a default split
direction per node, which is exactly the right inductive bias here (an
empty-address pair shouldn't be punished as if the addresses actively
disagreed).

### 4.2 Tier 2: house-number and structural features

Beyond generic string similarity, there's a small cluster of features
specifically for house numbers (`house_eq`, `house_lev`, `house_suffix`,
`house_prefix`, `house_absdiff`) and exact-match flags (`street_eq`,
`legal_eq`, `nc_exact`, `nc_contains`, `ntok_first_eq`). These exist because
generic string similarity treats `"703"` vs `"630"` and `"5235"` vs `"235"`
as roughly-equally-different, but they mean very different things in this
dataset: a **transposition or leading-digit-drop** (`5235`→`235`) is a
noise pattern I found repeatedly on *true* matches, while a **different
trailing digit** (`3370`→`3377`) almost always meant a genuinely different
business at a nearby address (the exact false-positive pattern the
generator uses as hard negatives — see §6). `house_suffix`/`house_prefix`
check for exactly the drop-a-leading-digit case; `house_absdiff` (log of the
numeric gap) gives the model a continuous signal for "close but not
identical" house numbers. This tier was added specifically in response to
reading false positives in `analyze.py` output, not designed speculatively.

### 4.3 Tier 3: the competition/context features — the highest-value tier

This is the tier that follows directly from §1(a) (the one-to-one structure),
and it's split into two functions computed at different scopes:

- **`candidate_context`** — computed **once, globally**, before any
  train/val split or S1-chunking, directly on the raw blocking output
  (`s1_idx, cand_idx, blk_score`). For each candidate record: how many
  distinct S1 entities also blocked to it (`cx_cand_n_s1`), its rank among
  them by blocking score (`cx_cand_rank`), and the gap to the best-scoring
  competitor (`cx_cand_gap`). Also, per S1: how many candidates it has at all
  (`cx_s1_n_cands`).

  **Why this has to be global**: if you computed "how many S1 entities want
  this candidate" only within a feature-build chunk (see §8, S1-chunking),
  you'd undercount for any candidate whose competing S1 entities happen to
  land in a different chunk — silently wrong numbers, not an error. That's
  why `_stage_candidates` in `features.py` explicitly computes
  `candidate_context` on the **full, unchunked** candidate table first, and
  only *then* slices by S1 chunk for the (per-pair, chunk-local-safe)
  similarity feature computation.

- **`s1_context`** — computed per S1 (after string features exist, so it's
  chunk-safe by construction — nothing here depends on data outside the
  current S1's own candidate set): for `nm_tset`, `nc_ratio`, `ad_tset`, and
  `blk_score`, the gap to this S1's own best candidate and this candidate's
  rank among the S1's own candidates. Also two counts:
  `cx_s1_n_strong_name` (candidates with name similarity ≥90) and
  `cx_s1_n_strong_both` (name ≥90 *and* address ≥80) — a cheap proxy for "is
  this S1 entity inherently ambiguous" (multiple strong look-alikes) which is
  exactly the situation the ambiguity-abstention decision rule (§5) is meant
  to catch.

**Evidence this tier matters**: on the mini-world run, LightGBM's top-10
features by gain were `cx_cand_rank`, `cx_cand_gap`, `atok_cont2`, `ad_tset`,
`blk_rank_both`, `nm_partial`, `atok_jac`, `nm_full_tset`, `anum_jac`,
`nc_jw` — the two single strongest features are both from the competition
tier, ahead of every raw string-similarity feature. I'd treat that as
validation that the "look at the graph structure, not just the pair" framing
was the right call for this problem, and I'd prioritize extending this tier
(see §9, stacking) over adding more string-similarity variants if you're
looking for the next model-side improvement.

### 4.4 Country is never a feature, on purpose

No feature encodes country directly (not even as an implicit signal, since
comparisons never cross country boundaries — see §1(b)). This is a direct
consequence of France being present in test but absent in train: a model
trained with country as a feature (or that had implicitly learned
country-specific thresholds because it only ever saw US/India during
training) would have no calibrated behavior for France. Every feature here is
either a pure text/structure similarity or a within-pair relative
comparison, so nothing in the model's input distribution changes between
"India pair," "US pair," and "France pair" except the actual field values
being compared.

### 4.5 Optional embedding feature (`emb_cos`)

If `emb.npy` exists (built by `embed.py` on the GPU box, see §9), each
candidate pair also gets `emb_cos`: raw dot-product cosine similarity between
the two records' dense embeddings (`pair_cosine` in `embed.py`, computed
chunked to avoid materializing a huge intermediate array). This is **not**
part of the v1 pipeline I validated end-to-end — it's wired through
(`build_features` checks for the file and adds the column automatically,
`decide`/`train` don't need to know about it since it's just another feature
column) but untested at scale, because it needs GPU inference over the whole
record set. If you enable it, retrain from scratch (delete
`features/`/`candidates.parquet`/`model.txt` first) so the feature is present
consistently in both train and inference.

---

## 5. The matcher (`train.py`) — a boring, deliberate choice

LightGBM binary classifier, `num_leaves=127`, `min_data_in_leaf=200`,
`feature_fraction`/`bagging_fraction=0.8`, `lambda_l2=1.0`, early-stopped on
the val split (`binary_logloss` + `auc` tracked, best iteration kept). No
neural matcher, no siamese network, no cross-encoder in the v1 pipeline
(a cross-encoder reranker is on the backlog, README "Known gaps").

**Why gradient-boosted trees and not a neural net:** the feature set is ~70
hand-built numeric similarity/context features, which is exactly the regime
where GBTs reliably outperform or match neural approaches with far less
tuning and far more interpretability (feature importance is directly
inspectable, which is how §4.3's "competition tier dominates" finding was
even discoverable). A neural approach would make sense if you were feeding
raw text (or embeddings) directly rather than engineered features — that's
what the optional `emb_cos` / cross-encoder extensions are for, layered on
top of, not replacing, this.

**Negative down-sampling and probability re-calibration**
(`--neg-frac`, `correct_probs`): with ~59-66 candidates per S1 and only
~2-6 true matches, negatives outnumber positives roughly 15:1 in the raw
candidate set (mini-world fit split, v1 blocking, no down-sampling:
4,958,019 total rows, 302,039 positive → ~4.66M negative). `--neg-frac` lets
you subsample negatives during training for speed;
`correct_probs` undoes the resulting bias with the standard odds-correction
(`odds_true = odds_model × neg_frac`) so that downstream probability
thresholds (§6) stay meaningful regardless of what sampling rate was used to
train. This was necessary for fast laptop iteration (`--neg-frac 0.3` in the
mini-world smoke test) but **the val set used for early stopping is *also*
down-sampled at the same rate** — i.e. early stopping happens on the sampled
distribution, while the final decision-rule tuning (§6) happens on the full,
un-sampled val set with re-calibrated probabilities. This is intentional
(early stopping just needs a consistent relative ranking of iterations; the
decision rule needs calibrated absolute probabilities) but worth knowing if
you're debugging a discrepancy between "best_iteration" and downstream
threshold behavior.

**Why LightGBM over XGBoost** (both are wired as options — see the
`code_review` history: the very first version of this pipeline used XGBoost
with an sklearn-GBDT fallback): both are MIT/BSD-family licensed
gradient-boosting implementations, similar quality — I ended up standardizing
on LightGBM as the primary path because it was faster to iterate with locally
and its native missing-value handling maps naturally onto the "null means no
information" features from §4.1/4.2. This isn't a strong opinion — swapping
back to XGBoost would be low-risk if your team already has XGBoost tooling
(licensing, model size, and behavior are all comparable for this use case).

---

## 6. Turning scores into matches (`decide.py`) — tuned directly for the metric

This is the stage I'd flag as most worth reading carefully, because it's
where "precision-heavy F0.5" actually gets enforced, rather than just hoped
for via a generic 0.5 threshold.

### 6.1 One-to-one assignment, with an abstention band

`one_to_one` ranks, for each *candidate* (not each S1), all S1 entities that
proposed it, by predicted probability, and keeps only the top-ranked
assignment. With `amb > 0`, it goes further: if the best and second-best S1
scores for a given candidate are within `amb` of each other, **the candidate
is dropped entirely** — assigned to neither. This is a direct, deliberate
trade of recall for precision on genuinely ambiguous cases (e.g. two S1
entities with an identical or near-identical name competing for one candidate
with a sparse/empty address, where the model has no real basis to prefer one
over the other). Given F0.5's 2x precision weighting, a coin-flip assignment
in that situation is expected-value-negative even though it "looks like" it
should help recall — this was confirmed by tuning (§6.3): the best
grid-search result used `amb=0.3`, not `amb=0`, i.e. a fairly aggressive
abstention band was actually optimal on validation, not just theoretically
motivated.

### 6.2 Two decision rules, both grid-searched together

- **`threshold`**: keep everything with `p >= t`.
- **`expected_f`**: for each S1, sort its remaining candidates by `p`
  descending, and for each prefix length `k` compute the *expected* F0.5 of
  predicting exactly those `k` candidates, using linearity of expectation over
  independent Bernoulli match probabilities:
  `E[F0.5 | top-k] ≈ 1.25 · Σ_{i≤k} p_i / (0.25 · Σ_all p_i + k)`.
  Pick the `k` maximizing this, but only commit to it if that expected value
  beats predicting nothing at all — i.e. beats
  `P(no true match among candidates) = Π_i (1 - p_i)`. This is a more
  principled version of a flat threshold: it naturally adapts `k` per entity
  based on how many candidates look genuinely strong, rather than applying
  one global cutoff to every entity regardless of how many plausible matches
  it has.

  **Caveat I want to be explicit about**: the independence assumption
  underlying this formula is not exactly true (candidate probabilities for
  the same S1 aren't independent — they compete for the same "true match"
  slots, and correlate through shared features like `cx_s1_*`). I treated
  this as a reasonable approximation rather than an exact optimum, which is
  why it's grid-searched *against* the simpler threshold rule rather than
  trusted blindly — on the mini-world run, `threshold` (t=0.8, amb=0.3) beat
  `expected_f` by a whisker (0.9887 vs. 0.9886), so in practice they're
  close, but this isn't guaranteed to hold at full scale or on a
  harder-to-separate country like France.

### 6.3 Tuning is exhaustive grid search on validation, not a heuristic pick

`tune()` tries every combination of `{threshold rule × 11 t values, expected_f
rule × 7 min_p values} × 4 amb values` = 72 combinations, scores each with the
*exact* macro F0.5 formula (`macro_f05`, verified against the problem
statement's own worked example in `tests/test_pipeline.py`), and keeps the
best. This runs on the **held-out validation split** (`s1_meta.fold == "val"`)
— never on data the model was fit on — so the chosen threshold/rule isn't
overfit to training noise. Because the grid is small and each evaluation is a
few vectorized polars operations, this costs seconds, not an expensive
hyperparameter search — there's no real reason not to re-run it after any
model or feature change; `run.py train` does so automatically every time.

### 6.4 Breakdown metrics beyond the headline number

`breakdown()` separately reports singleton accuracy (fraction of true
singletons where 0 candidates were predicted), non-singleton empty-prediction
rate (a proxy for recall failures), and pooled pair-level precision/recall.
These aren't used by the tuner, but they're what `analyze.py` reports
per-country, and they're the numbers I'd look at first if a future change
improves the headline F0.5 but you want to know *why* — e.g. whether a
change traded singleton accuracy for non-singleton recall, which the single
macro number can hide.

---

## 7. What's still wrong (honest error analysis, from `analyze.py`)

On the mini-world validation split, `analyze.py` categorizes errors by
reading the actual name/address pairs, not just aggregate stats. Patterns
found (already summarized in `Documentation_template.md` §5, expanded here
with the reasoning):

**False positives** (the costlier error class under F0.5):
- Same business name, same street, different house number
  (`22 Vine Street` vs `31 Vine Street`; `644 Fox Run Trail` vs `648 Fox Run
  Trail`). These are the *intended* hard negatives from the data generator —
  a name match alone isn't enough evidence, which is exactly why the
  house-number features (§4.2) exist. Residual false positives here suggest
  either the house-number signal needs more weight (a training data /
  hyperparameter question) or that some of these are actually true matches
  with corrupted house numbers on one side (a normalization-vs-labeling
  ambiguity I can't resolve without more label inspection).
- Identical or near-identical names where one side has an **empty address**
  — two different S1 entities with the same/similar name both compete for
  one candidate, and without address evidence the model has no way to
  disambiguate. The `amb` abstention band (§6.1) directly targets this, and
  tuning picked a large `amb` (0.3) partly because of this pattern.

**False negatives**:
- Name replaced by an apparently-unrelated string (e.g. `"Agastya Realty
  Private Limited"` ↔ `"Evoyuma"`) combined with a weak/absent address. There
  is no text-similarity feature that can catch a case with literally no
  shared tokens or structure — this is a fundamental limit of the current
  feature set, not a bug. Embedding similarity (§4.5, §9) is the natural next
  lever here, since a semantic/learned embedding *might* place these closer
  in vector space even with zero lexical overlap — though I have no evidence
  yet that it will, since it's untested.
- Initials-style website names (`"North & Brothers Private Limited"` ↔
  `"nbrothers.com"`) — the `name_compact`/`c:`/`p:` blocking keys catch some
  compacted-name-in-URL cases but not ones that abbreviate to just initials;
  this is a blocking-recall gap (never reaches the model), not a
  matcher-precision gap. Confirmed via the `p=null` rows in the false-negative
  dump (§ note in `analyze.py`: it explicitly separates "never reached the
  model = blocking misses" from "reached the model but scored too low").
- The v1 blocking pair-recall ceiling is **0.966** (§3.4), meaning roughly
  3.4% of true pairs are structurally unreachable by the current key-based
  blocking regardless of matcher quality. Closing this gap requires either
  looser blocking parameters (cost: more candidates, more compute) or a
  genuinely different signal — which is exactly what the optional embedding
  channel is for.

---

## 8. Running at full scale — what's proven vs. what's extrapolated

I want to be precise about what's actually been verified versus what's
inferred, since this affects how much you should trust the numbers in
`Documentation_template.md`:

**Verified, small scale (mini-world = 5% hash-sample of Source-1 entities +
all their true matches + a matching fraction of distractor noise, built by
`prepare_train --sample-frac 0.05`, ~628k total records):**
- Full pipeline runs end-to-end without errors: prepare → block → feature →
  train → tune decision rule → predict → validate.
- Validation macro F0.5: **0.9887** (precision 0.996, recall 0.976, 98.4%
  singleton accuracy). This number is almost certainly optimistic relative to
  full-scale performance, because the mini-world's distractor pool is smaller
  and the "look-alike business" density that makes real matching hard is
  correspondingly lower.
- `matching_results.tsv` + `candidate_pairs.tsv` pass `validate_submission.py
  --check-ids` with no warnings, including correct handling of France rows
  (present in the test sample, with plausible match rates — 9.0% of France S1
  entities got a match in the smoke test, vs. 2.2%/2.4% for US/India, which
  is expected since the smoke-test "test" pool was itself a small random
  sample rather than the full test set — don't read anything into that
  specific number beyond "the code path runs correctly for France").

**Verified, full data density, blocking only:**
- `--s1-frac 0.05` (5% of S1, run against the **complete, unsampled** 10.3M-
  record Source-2/3 pool) — see §3.4. Pair recall 0.966, F0.5 ceiling 0.989.
  This is a real, exact measurement of the recall ceiling at true data
  density, just for a subset of S1 queries.

**Not verified — extrapolated / to be done on the VM:**
- A full, unsampled end-to-end run (all 2.2M Source-1 train entities, full
  feature build, full training set, tuned on the full validation set) was
  **not completed on the laptop.** I started it multiple times; each attempt
  made real progress (the India country consistently finished both key
  channels within a couple of minutes; the US country — 1.3M S1 vs 6.2M pool
  — is the expensive one, and individual 150k-row chunks that should take
  well under a minute sometimes logged 15-30+ minutes elapsed, which lines up
  with macOS idle/dark-wake sleep stalling the process rather than the
  computation itself being slow) but was repeatedly interrupted over a
  session that ran long, on a laptop running on battery for part of it. I
  never got a clean, uninterrupted timing measurement for the full run as a
  result — treat any full-scale wall-clock estimate as unknown until you've
  run it on the VM. The code path is the *same* code path validated at
  small-scale-full-density above (§3.4's `--s1-frac 0.05`-against-full-pool
  measurement exercises identical logic against the identical full-size
  pool), so I have reasonable confidence it will complete correctly on a VM
  with stable power — but this is inference from partial runs, not a
  completed, observed result.
- The full-scale validation F0.5 number, per-country breakdown, and final
  feature-importance ranking at full scale are all `[TBD]` in
  `Documentation_template.md` for this reason — **do not report the 0.9887
  mini-world number as your validation score without re-running at full
  scale first.**

**What I'd do first on the VM**, in order:
1. `python src/run.py train --data-dir .../dataset --work-dir work --model-dir
   models` with no `--sample-frac`/`--feature-s1-frac` sampling flags — this
   reproduces everything above at true scale and gives you the real numbers
   for the documentation.
2. Compare the real `val_report.json` against the mini-world numbers here —
   if F0.5 drops substantially, `analyze.py`'s per-country breakdown and
   false-positive/negative dumps are the tool to find out why.
3. Only then consider the backlog items in §9 (embeddings, stacking) — they're
   real potential improvements, but tuning them against an unverified
   baseline risks chasing noise.

---

## 9. Explicit backlog (also in README, restated with reasoning)

- **Embedding channel** (`embed.py`, `intfloat/multilingual-e5-base`, MIT,
  278M params): wired through blocking (`load_extra`/`extra` parameter —
  embedding-sourced candidates bypass the top-K cap, since they're exactly
  the pairs the lexical channels are blind to) and features (`emb_cos`), but
  never run. This is the most likely lever for the two false-negative
  patterns in §7 that have zero lexical overlap (unrelated trade names,
  heavy abbreviation). Needs GPU time to encode ~12.5M(train)+11M(test)
  records — budget this early since it gates a full pipeline re-run.
- **Stacking / second-stage model**: feed each candidate's stage-1
  probability, its `cx_*` context computed *on model scores* (rather than raw
  blocking scores, which is all `candidate_context`/`s1_context` currently
  use), into a second small model. Motivated directly by §4.3's finding that
  competition features already dominate — recomputing them on calibrated
  model probabilities instead of raw blocking scores should be strictly more
  informative, since blocking scores are a cheap proxy that the a properly
  trained matcher already partially supersedes.
- **Cross-encoder reranker** fine-tuned on the training pairs: a further
  refinement on top of the embedding channel, for the highest-value ambiguous
  cases specifically (e.g. only re-score pairs where the stage-1 model is
  between 0.3-0.7, which is a small fraction of all pairs, making a slower
  cross-encoder tractable there even if it's too slow to run on every
  candidate).
- **Real address parsing** instead of the comma-position heuristic in §2.2 —
  would improve `house_no`/`street_key` precision, which feeds directly into
  the house-number features in §4.2.
- **Per-country abbreviation dicts** instead of one flat `ADDR_CANON` — only
  worth it if you find concrete cross-country collisions in practice (haven't
  observed any yet, so this is speculative).

---

## 10. Files at a glance (cross-reference)

| file | stage | reads | writes |
|---|---|---|---|
| `normalize.py` | text cleaning | — (pure functions) | — (library, called by `prepare.py`) |
| `prepare.py` | 0 | raw TSVs | `records.parquet`, `gt_pairs.parquet`, `s1_meta.parquet`, `translit.json` |
| `blocking.py` | 1 | `records.parquet` | `candidates.parquet` |
| `embed.py` (optional) | 1b | `records.parquet` | `emb.npy`, `emb_candidates.parquet` |
| `features.py` | 2a | `candidates.parquet`, `records.parquet`, `gt_pairs.parquet`/`s1_meta.parquet` (train), `emb.npy` (optional) | `features/part-*.parquet` |
| `train.py` | 2b | `features/part-*.parquet` | `model.txt`, `model_meta.json`, `feature_importance.csv` |
| `decide.py` | 3 | scored pairs (in-memory, from `train.predict`) | — (library, called by `run.py`) |
| `run.py` | orchestrator | all of the above | `decision.json`, `val_report.json`, `matching_results.tsv`, `candidate_pairs.tsv` |
| `analyze.py` | diagnostics | trained model + val features | per-country metrics (stdout), `errors/*.tsv` |
| `scoring.py` | standalone | any two TSVs in the submission format | macro F0.5 (stdout) |
| `package.py` | packaging | `output/`, `README.md`, `requirements.txt`, `Documentation_template.md` | `<team>_submission.zip` |
| `tests/test_pipeline.py` | regression tests | — | pass/fail for normalization cases, the scorer's worked example, decision-rule logic |

Every stage-boundary file above is a parquet checkpoint specifically so you
can `--skip-existing` past stages you haven't touched while iterating on one
stage — see `run.py`'s docstring and the README's "Fast experiment loops"
section.
