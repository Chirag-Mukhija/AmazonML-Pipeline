"""Fast regression checks. Run: python tests/test_pipeline.py  (or pytest tests/)"""
import os
import sys
import tempfile

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import decide  # noqa: E402
from normalize import learn_translit_dict, normalize_frame, skeleton  # noqa: E402
from scoring import score, score_entity  # noqa: E402


def _norm(name, addr, translit=None):
    df = pl.DataFrame({"entity_id": ["x"], "business_name": [name], "business_address": [addr], "country": ["US"]})
    return normalize_frame(df, translit).row(0, named=True)


def test_name_normalization():
    r = _norm("desertcapitalkkr.com", "")
    assert r["name_compact"] == "desertcapitalkkr" and r["name_website"]
    assert _norm("Desert Capital Kkr Inc.", "")["name_compact"] == "desertcapitalkkr"
    r = _norm("Ecstatic Ecstatic Desh Private (Limited)", "")
    assert r["name_core"] == "ecstatic desh" and r["legal_form"] == "ltd pvt"
    assert _norm("Rama Sartin + Inc", "")["name_core"] == _norm("Rama & Sartin Inc", "")["name_core"]
    assert _norm("Sequoia Inc Sérvice", "")["name_core"] == "sequoia service"
    assert _norm("Établissements Salsa SAS", "")["legal_form"] == "sas"
    # OCR / leetspeak noise, dotted legal forms, ids and phone numbers
    assert _norm("5TAY & BROTHERS LIMITED", "")["name_core"] == "stay brothers"
    assert _norm("Satya F0undation", "")["name_core"] == "satya foundation"
    assert _norm("A Cure 4 IT LLC", "")["name_core"] == "a cure 4 it"  # standalone digits untouched
    r = _norm("ZAA Energy L.L.C. (ID: 29782)", "")
    assert r["name_core"] == "zaa energy" and r["legal_form"] == "llc"
    assert _norm("F 8 Prime - 6148769058", "")["name_core"] == "f 8 prime"
    assert _norm("lnterstate Systems", "")["name_skel"] == _norm("Interstate Systems", "")["name_skel"]


def test_address_normalization():
    r = _norm("x", "Tallassee, 00703 Powers Extension, AL")
    assert r["house_no"] == "703" and r["street_key"] == "703_powers"
    assert "ext" in r["addr_tokens"]
    a = _norm("x", "B-11/2, Garima Garden, Ghaziabad")
    b = _norm("x", "B-1-1/2, GARIMA GARDEN, GHAZIABAD")
    assert "b112" in a["addr_ids"] and "b112" in b["addr_ids"]
    assert _norm("x", "405 Dunlop Dr, Opelika, Alabama, <NULL>")["addr_norm"].endswith("alabama")
    assert _norm("x", "12 R. Calvé, Bordeaux")["addr_tokens"][:2] == ["12", "rue"]
    assert _norm("x", "")["addr_empty"]


def test_skeleton_and_translit():
    assert skeleton("haelhtacre") == skeleton("healthcare")
    pairs = pl.DataFrame({"n1": ["Vision Products Private Limited"] * 2 + ["Ram Marketing Private Limited"],
                          "n2": ["विजन प्रोडक्ट्स प्राइवेट लिमिटेड"] * 2 + ["राम मार्केटिंग प्राइवेट लिमिटेड"]})
    tr = learn_translit_dict(pairs, min_count=1)
    assert tr["प्राइवेट"] == "private" and tr["विजन"] == "vision"
    assert _norm("विजन प्रोडक्ट्स प्राइवेट लिमिटेड", "", tr)["name_core"] == "vision products"


def test_scorer_matches_problem_statement_example():
    assert abs(score_entity({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"}) - 0.714) < 1e-3
    assert score_entity(set(), set()) == 1.0
    assert score_entity({"S2-1"}, set()) == 0.0
    assert score_entity(set(), {"S2-1"}) == 0.0
    with tempfile.TemporaryDirectory() as d:
        pred, truth = os.path.join(d, "p.tsv"), os.path.join(d, "t.tsv")
        with open(pred, "w") as f:
            f.write("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S2-2\nS1-2\t\n")
        with open(truth, "w") as f:
            f.write("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\nS1-2\t\n")
        assert abs(score(pred, truth, verbose=False) - (1.25 * 0.5 / (0.25 * 0.5 + 1) + 1) / 2) < 1e-9


def test_decision_rules():
    scored = pl.DataFrame({
        "s1_idx": [1, 1, 2, 3], "cand_idx": [10, 11, 10, 12],
        "p": [0.9, 0.2, 0.6, 0.95], "label": [1, 0, 0, 1],
    })
    one = decide.one_to_one(scored)
    assert set(one.filter(pl.col("cand_idx") == 10)["s1_idx"].to_list()) == {1}  # cand 10 -> best S1 only
    pred = decide.apply_rule(scored, {"rule": "threshold", "t": 0.5})
    assert sorted(pred["cand_idx"].to_list()) == [10, 12]
    meta = pl.DataFrame({"s1_idx": [1, 2, 3, 4], "n_true": [1, 0, 1, 0]})
    assert decide.macro_f05(pred, meta) == 1.0
    ef = decide.apply_rule(scored, {"rule": "expected_f", "min_p": 0.1})
    assert 11 not in ef["cand_idx"].to_list()
    # ambiguity abstention: candidate 20 is a near-tie between S1 5 and 6 -> nobody gets it
    tie = pl.DataFrame({"s1_idx": [5, 6, 7], "cand_idx": [20, 20, 21], "p": [0.91, 0.90, 0.9], "label": [1, 0, 1]})
    assert decide.one_to_one(tie, amb=0.05)["cand_idx"].to_list() == [21]
    assert sorted(decide.one_to_one(tie, amb=0.0)["s1_idx"].to_list()) == [5, 7]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
