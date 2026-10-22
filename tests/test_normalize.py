from graphrag.extraction.normalize import clean_surface, entity_key, relation_key


def test_case_variants_share_a_key():
    assert entity_key("Aspirin") == entity_key("aspirin") == entity_key("ASPIRIN")


def test_whitespace_quotes_punctuation_and_articles():
    assert entity_key('  "The   Aspirin." ') == "aspirin"


def test_unicode_dashes_unified():
    assert entity_key("COX\u20111") == entity_key("COX-1") == "cox-1"


def test_bare_article_is_not_stripped_to_empty():
    assert entity_key("A") == "a"


def test_display_form_keeps_case():
    assert clean_surface("  Bayer   AG ") == "Bayer AG"


def test_relation_key():
    assert relation_key("Is_Located  In") == "is located in"
    assert relation_key("INHIBITS.") == "inhibits"
