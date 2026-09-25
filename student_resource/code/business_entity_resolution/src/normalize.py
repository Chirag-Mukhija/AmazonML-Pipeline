"""Name/address normalization, vectorized with polars.

Only the records' own text plus small static token tables are used (legal
forms, street-type abbreviations in English/French/Indian usage). No external
lookups. Non-ASCII text (Indic scripts, French accents) is transliterated with
`unidecode`, which ships a static character table.

Output columns added by `normalize_frame` (all lowercase ASCII):
  name_norm      cleaned full name (legal forms kept)
  name_core      name without legal-form / filler tokens, deduped tokens
  name_compact   name_core without spaces (matches "desertcapitalkkr.com")
  name_tokens    list[str] of core tokens
  name_skel      list[str] consonant skeletons of core tokens (typo/translit tolerant)
  legal_form     canonical legal form ("" if none)
  name_nonlatin  bool, original name had Indic/other non-Latin script
  name_website   bool, original name looked like a URL
  addr_norm      cleaned address
  addr_tokens    list[str] canonical address tokens (abbreviations unified)
  addr_numbers   list[str] pure-number tokens, leading zeros stripped
  addr_ids       list[str] compact alphanumeric identifiers ("b-11/2" -> "b112")
  house_no       first number in the first comma-component that has a digit
  street_key     house_no + first alpha token after it ("703_powers")
  addr_empty     bool
"""
import polars as pl
from unidecode import unidecode

NON_ASCII_RE = r"[^\x00-\x7F]"
NON_LATIN_RE = (r"[\p{Devanagari}\p{Tamil}\p{Telugu}\p{Gurmukhi}\p{Bengali}\p{Gujarati}"
                r"\p{Kannada}\p{Malayalam}\p{Oriya}\p{Arabic}\p{Cyrillic}\p{Han}]")

# Canonical legal form for each surface variant. Removed from name_core.
LEGAL_FORMS = {
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "llc": "llc", "llp": "llp", "lp": "lp",
    "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt", "plc": "plc",
    "pllc": "pllc", "pc": "pc", "opc": "opc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl",
    "snc": "snc", "sci": "sci", "scop": "scop", "selarl": "selarl",
    "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv",
}
# Words the noise generator sprinkles in / that carry little identity.
FILLER = {"the", "and", "of", "dba", "www", "com", "shri", "sri", "dr", "m", "s"}

ADDR_CANON = {
    # English street types / units
    "street": "st", "str": "st", "avenue": "ave", "av": "ave", "avn": "ave",
    "boulevard": "blvd", "bd": "blvd", "boul": "blvd", "road": "rd", "drive": "dr",
    "lane": "ln", "court": "ct", "circle": "cir", "place": "pl", "square": "sq",
    "highway": "hwy", "parkway": "pkwy", "trail": "trl", "extension": "ext",
    "apartment": "apt", "appartement": "apt", "suite": "ste", "building": "bldg",
    "floor": "fl", "flr": "fl", "north": "n", "south": "s", "east": "e", "west": "w",
    "mount": "mt", "saint": "st", "sainte": "ste", "fort": "ft", "point": "pt",
    "township": "twp", "county": "cnty", "number": "no", "nos": "no",
    # French
    "rue": "rue", "r": "rue", "allee": "all", "impasse": "imp", "chemin": "chem",
    "route": "rte", "faubourg": "fbg", "quai": "qu", "residence": "res",
    # India
    "near": "nr", "opposite": "opp", "opposit": "opp", "behind": "bh", "nagar": "ngr",
    "marg": "marg", "colony": "col", "sector": "sec", "village": "vil", "post": "po",
    "district": "dist", "dist": "dist",
}
# Pure noise tokens that carry no location identity.
ADDR_DROP = {"null", "na", "n", "a", "unit", "the", "of", "de", "du", "des", "la", "le", "les"}

_VOWELS = str.maketrans("", "", "aeiouyh")


def _skeleton(token: str) -> str:
    """Consonant skeleton: tolerant to vowel typos and Indic transliteration."""
    t = token.replace("ph", "f").replace("ck", "k").replace("x", "ks")
    t = t.translate(str.maketrans({"c": "k", "q": "k", "z": "s", "w": "v", "i": "l"}))
    if not t:
        return ""
    head, tail = t[0], t[1:].translate(_VOWELS)
    out = [head]
    for ch in tail:
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)


def skeleton_expr(e: pl.Expr) -> pl.Expr:
    """Vectorized `_skeleton` (same rules) for polars string columns."""
    e = (e.str.replace_all("ph", "f").str.replace_all("ck", "k").str.replace_all("x", "ks")
         .str.replace_all("[cq]", "k").str.replace_all("z", "s").str.replace_all("w", "v")
         .str.replace_all("i", "l"))  # "lnterstate"/"lT" OCR confusion: fold i and l together
    e = e.str.slice(0, 1) + e.str.slice(1).str.replace_all("[aeiouyh]", "")
    for ch in "bdfgjklmnprstv0123456789":
        e = e.str.replace_all(f"{ch}{ch}+", ch)
    return e


_LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b"}


def _deleet(e: pl.Expr) -> pl.Expr:
    """OCR/leetspeak noise: a single digit embedded in a word ("5tay", "youn6",
    "f0undation") becomes the letter it imitates. Pure numbers are untouched."""
    is_leet = e.str.contains(r"^[a-z]*[013-8][a-z]*$") & (e.str.count_matches(r"[a-z]") >= 3)
    fixed = e
    for d, ch in _LEET.items():
        fixed = fixed.str.replace_all(d, ch, literal=True)
    return pl.when(is_leet).then(fixed).otherwise(e)


def ascii_fold(s: pl.Series) -> pl.Series:
    """unidecode only the rows that contain non-ASCII (a minority)."""
    mask = s.str.contains(NON_ASCII_RE).fill_null(False)
    if not mask.any():
        return s
    idx = mask.arg_true()
    folded = pl.Series([unidecode(x) for x in s.filter(mask).to_list()], dtype=pl.String)
    return s.scatter(idx, folded)


def _apply_translit_dict(names: pl.Series, translit: dict) -> pl.Series:
    """Map known non-Latin tokens to their learned Latin equivalents."""
    return pl.Series([" ".join(translit.get(t, t) for t in s.split()) for s in names.to_list()],
                     dtype=pl.String)


def normalize_frame(df: pl.DataFrame, translit: dict = None) -> pl.DataFrame:
    """df has entity_id, business_name, business_address, country."""
    name = pl.col("business_name").fill_null("")

    df = df.with_columns(
        name.str.contains(NON_LATIN_RE).alias("name_nonlatin"),
        name.str.contains(r"(?i)(www\.|\.com\b|\.net\b|\.org\b|\.co\.in\b|\.in\b|\.fr\b)").alias("name_website"),
    )

    raw_name = df["business_name"].fill_null("").str.to_lowercase()
    if translit and df["name_nonlatin"].any():
        mask = df["name_nonlatin"]
        fixed = _apply_translit_dict(raw_name.filter(mask), translit)
        raw_name = raw_name.scatter(mask.arg_true(), fixed)
    df = df.with_columns(ascii_fold(raw_name).alias("_name"))

    df = df.with_columns(
        pl.col("_name")
        .str.to_lowercase()
        .str.replace_all(r"www\.", " ")
        .str.replace_all(r"\.(com|net|org|co\.in|in|fr|biz|info|us)\b", " ")
        .str.replace_all(r"d/b/a|\bdba\b", " ")
        .str.replace_all(r"\(?\bid\s*[:#]?\s*\d+\)?", " ")          # "(ID: 29782)"
        .str.replace_all(r"\b([a-z])\.([a-z])\.([a-z])\.?", "$1$2$3")  # l.l.c. -> llc
        .str.replace_all(r"\b([a-z])\.([a-z])\.?(\s|$)", "$1$2$3")     # p.c. -> pc
        .str.replace_all(r"\s\+\s|&", " and ")
        .str.replace_all(r"[^a-z0-9]+", " ")
        .str.replace_all(r"\b\d{7,}\b", " ")                           # phone numbers
        .str.strip_chars()
        .alias("name_norm")
    )

    legal_keys = list(LEGAL_FORMS)
    df = df.with_columns(pl.col("name_norm").str.split(" ").list.eval(_deleet(pl.element())).alias("_ntok"))
    df = df.with_columns(pl.col("_ntok").list.join(" ").alias("name_norm"))
    df = df.with_columns(
        pl.col("_ntok").list.eval(
            pl.element().filter(pl.element().is_in(legal_keys)).replace_strict(LEGAL_FORMS, default="")
        ).list.unique(maintain_order=True).list.sort().list.join(" ").alias("legal_form"),
        pl.col("_ntok").list.eval(
            pl.element().filter(
                ~pl.element().is_in(legal_keys) & ~pl.element().is_in(list(FILLER)) & (pl.element() != "")
            )
        ).list.unique(maintain_order=True).alias("name_tokens"),
    )
    df = df.with_columns(
        pl.col("name_tokens").list.join(" ").alias("name_core"),
        pl.col("name_tokens").list.join("").alias("name_compact"),
    )
    # Names made only of legal/filler words: fall back to the full normalized name.
    df = df.with_columns(
        pl.when(pl.col("name_core") == "").then(pl.col("name_norm")).otherwise(pl.col("name_core")).alias("name_core"),
        pl.when(pl.col("name_compact") == "").then(pl.col("name_norm").str.replace_all(" ", "")).otherwise(pl.col("name_compact")).alias("name_compact"),
    )
    df = df.with_columns(
        pl.col("name_tokens").list.eval(skeleton_expr(pl.element())).alias("name_skel")
    )

    # ---------------- address ----------------
    df = df.with_columns(ascii_fold(df["business_address"].fill_null("")).alias("_addr"))
    df = df.with_columns(
        pl.col("_addr").str.to_lowercase()
        .str.replace_all(r"<\s*null\s*>|\bnull\b", " ")
        .alias("_addr")
    )
    df = df.with_columns(
        pl.col("_addr").str.replace_all(r"[^a-z0-9,]+", " ").str.replace_all(r"\s*,\s*", ", ")
        .str.replace_all(r"\s+", " ").str.strip_chars(" ,").alias("addr_norm"),
    )
    df = df.with_columns((pl.col("addr_norm").str.len_chars() == 0).alias("addr_empty"))

    canon_keys = list(ADDR_CANON)
    tok = (
        pl.col("addr_norm").str.replace_all(",", " ").str.split(" ")
        .list.eval(
            pl.element()
            .str.replace(r"^(\d+)(st|nd|rd|th)$", "$1")
            .str.replace(r"^0+(\d)", "$1")
        )
    )
    df = df.with_columns(tok.alias("_atok"))
    df = df.with_columns(
        pl.col("_atok").list.eval(
            pl.when(pl.element().is_in(canon_keys))
            .then(pl.element().replace_strict(ADDR_CANON, default=None))
            .otherwise(pl.element())
        ).list.eval(
            pl.element().filter((pl.element() != "") & ~pl.element().is_in(list(ADDR_DROP)))
        ).alias("addr_tokens")
    )
    df = df.with_columns(
        pl.col("addr_tokens").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$")))
        .list.unique(maintain_order=True).alias("addr_numbers"),
        # alphanumeric identifiers, e.g. "B-11/2" and "B-1-1/2" both -> "b112"
        pl.col("_addr").str.replace_all(r"[\s,;]+", " ").str.split(" ")
        .list.eval(
            pl.element().filter(pl.element().str.contains(r"\d") & pl.element().str.contains(r"[a-z/\-\.]"))
            .str.replace_all(r"[^a-z0-9]", "")
            .str.replace(r"^0+(\d)", "$1")
        ).list.eval(pl.element().filter(pl.element().str.len_chars() >= 2))
        .list.unique(maintain_order=True).alias("addr_ids"),
    )

    # House number / street key from the first comma-component containing a digit.
    comp = (
        pl.col("addr_norm").str.split(", ")
        .list.eval(pl.element().filter(pl.element().str.contains(r"\d")))
        .list.first().fill_null("")
    )
    df = df.with_columns(comp.alias("_street_comp"))
    df = df.with_columns(
        pl.col("_street_comp").str.extract(r"(\d+)", 1).str.replace(r"^0+(\d)", "$1").fill_null("").alias("house_no"),
        pl.col("_street_comp").str.extract(r"\d+\w*\s+([a-z]{2,})", 1).fill_null("").alias("_street_word"),
    )
    df = df.with_columns(
        pl.when((pl.col("house_no") != "") & (pl.col("_street_word") != ""))
        .then(pl.col("house_no") + "_" + pl.col("_street_word"))
        .otherwise(pl.lit(""))
        .alias("street_key")
    )

    return df.drop(["_name", "_ntok", "_addr", "_atok", "_street_comp", "_street_word"])


def skeleton(token: str) -> str:
    return _skeleton(token)


def learn_translit_dict(pairs: pl.DataFrame, min_count: int = 2, min_share: float = 0.5) -> dict:
    """Learn non-Latin-token -> Latin-token mapping from labelled training pairs.

    `pairs` has columns n1 (Source-1 name, Latin) and n2 (Source-2/3 name).
    Uses only pairs where n2 is non-Latin and both names have the same
    token count, aligning tokens positionally. Purely data-driven -- this is
    how we get "प्राइवेट" -> "private" without any external dictionary.
    """
    sub = pairs.filter(pl.col("n2").str.contains(NON_LATIN_RE))
    sub = sub.with_columns(
        pl.col("n1").str.to_lowercase().str.replace_all(r"[^\w\s]", " ").str.split(" ")
        .list.eval(pl.element().filter(pl.element() != "")).alias("t1"),
        pl.col("n2").str.to_lowercase().str.replace_all(r"[^\w\s]", " ").str.split(" ")
        .list.eval(pl.element().filter(pl.element() != "")).alias("t2"),
    ).filter(pl.col("t1").list.len() == pl.col("t2").list.len())
    aligned = sub.select(pl.col("t2").alias("src"), pl.col("t1").alias("dst")).explode(["src", "dst"])
    aligned = aligned.filter(pl.col("src").str.contains(NON_LATIN_RE))
    counts = aligned.group_by(["src", "dst"]).len()
    totals = counts.group_by("src").agg(pl.col("len").sum().alias("total"))
    best = (
        counts.sort("len", descending=True).group_by("src").first()
        .join(totals, on="src")
        .filter((pl.col("len") >= min_count) & (pl.col("len") / pl.col("total") >= min_share))
    )
    return dict(zip(best["src"].to_list(), best["dst"].to_list()))
