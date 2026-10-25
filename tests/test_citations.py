from graphrag.generation.citations import canonicalize, check_answer, extract_citations

IDS = {"zoraxin:000", "tremidol:000", "halvern-therapeutics:000"}


def test_bracket_styles_all_canonicalize():
    raw = (
        "A [zoraxin:000]. B (tremidol:000). C 【halvern-therapeutics:000】. "
        "D [zoraxin:000, tremidol:000]. E tremidol:000. F [Zoraxin : 000]."
    )
    text = canonicalize(raw)
    assert "(tremidol:000)" not in text and "【" not in text
    assert "[zoraxin:000][tremidol:000]" in text
    assert "E [tremidol:000]." in text
    assert "F [zoraxin:000]." in text
    assert extract_citations(text) == ["zoraxin:000", "tremidol:000", "halvern-therapeutics:000"]


def test_times_and_plain_brackets_are_not_citations():
    text = canonicalize("Dosed at 12:300 and 10:30 [see label] (twice daily).")
    assert extract_citations(text) == []


def test_valid_answer_accepted():
    g = check_answer("Tremidol inhibits Velkase-3 [tremidol:000].", IDS)
    assert (g.status, g.citations) == ("answered", ["tremidol:000"])


def test_unretrieved_citation_rejected():
    g = check_answer("Tremidol raises it [tremidol:000][brimlase:000].", IDS)
    assert g.status == "rejected"
    assert g.invalid_citations == ["brimlase:000"]
    assert g.citations == ["tremidol:000"]


def test_uncited_answer_rejected():
    g = check_answer("Tremidol raises Zoraxin levels.", IDS)
    assert (g.status, g.reason) == ("rejected", "answer has no citations")


def test_refusal_detected_in_any_form():
    variants = ("INSUFFICIENT_EVIDENCE", "insufficient_evidence.", "INSUFFICIENT_EVIDENCE: no")
    for text in variants:
        assert check_answer(text, IDS).status == "insufficient_evidence"
