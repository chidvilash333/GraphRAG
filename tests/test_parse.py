import json

from graphrag.extraction.parse import parse_triples


def _raw(*items):
    return json.dumps({"triples": list(items)})


def _t(s, r, o):
    return {"subject": s, "relation": r, "object": o}


def test_valid_triples_parsed_and_normalized():
    res = parse_triples(_raw(_t("Aspirin", "Inhibits", "COX-1")), max_triples=10)
    assert res.error is None
    (t,) = res.triples
    assert (t.subject, t.relation, t.object) == ("Aspirin", "inhibits", "COX-1")
    assert t.key == ("aspirin", "inhibits", "cox-1")


def test_malformed_items_dropped_with_reasons():
    raw = _raw(
        _t("Aspirin", "inhibits", "COX-1"),
        "not a dict",
        {"subject": "Aspirin", "relation": "inhibits"},  # missing object
        {"subject": 1, "relation": "inhibits", "object": "X"},  # non-string
        _t("  ", "inhibits", "COX-1"),  # empty
        _t("Aspirin", "inhibits", "aspirin"),  # self-loop
        _t("It", "inhibits", "COX-2"),  # pronoun
        _t("X" * 200, "inhibits", "COX-1"),  # too long
    )
    res = parse_triples(raw, max_triples=10)
    assert len(res.triples) == 1
    assert [reason for _, reason in res.dropped] == [
        "not an object",
        "missing or non-string field",
        "missing or non-string field",
        "empty after normalization",
        "self-loop",
        "pronoun entity",
        "entity too long",
    ]


def test_case_duplicates_collapse():
    raw = _raw(_t("Aspirin", "inhibits", "COX-1"), _t("aspirin", "INHIBITS", "cox-1"))
    res = parse_triples(raw, max_triples=10)
    assert len(res.triples) == 1
    assert res.dropped[0][1] == "duplicate"


def test_max_triples_enforced():
    raw = _raw(*[_t(f"Drug {i}", "inhibits", "COX-1") for i in range(5)])
    res = parse_triples(raw, max_triples=3)
    assert len(res.triples) == 3
    assert sum(r == "over max_triples" for _, r in res.dropped) == 2


def test_code_fences_tolerated():
    raw = "```json\n" + _raw(_t("Aspirin", "inhibits", "COX-1")) + "\n```"
    assert len(parse_triples(raw, max_triples=10).triples) == 1


def test_invalid_json_sets_error_not_exception():
    res = parse_triples("sorry, I can't", max_triples=10)
    assert res.triples == [] and res.error is not None


def test_wrong_top_level_shape():
    assert parse_triples('{"edges": []}', max_triples=10).error is not None
    assert parse_triples("[1, 2]", max_triples=10).error is not None


def test_grounding_drops_hallucinated_entities():
    text = "Aspirin inhibits COX-1."
    raw = _raw(_t("Aspirin", "inhibits", "COX-1"), _t("Aspirin", "treats", "Migraine"))
    res = parse_triples(raw, max_triples=10, source_text=text)
    assert [t.object for t in res.triples] == ["COX-1"]
    assert res.dropped[0][1] == "entity not found in source text"
