from graphrag.ingest.chunking import chunk_document
from graphrag.ingest.documents import Document
from helpers import SubwordTokenizer, WhitespaceTokenizer

TEXT = " ".join(f"Word{i}" for i in range(25))


def _chunks(text=TEXT, tok=None, max_tokens=10, overlap=3):
    doc = Document("doc", "doc.md", text)
    return chunk_document(
        doc, tok or WhitespaceTokenizer(), max_tokens=max_tokens, overlap_tokens=overlap
    )


def test_windows_respect_size_and_overlap():
    chunks = _chunks()
    assert all(c.n_tokens <= 10 for c in chunks)
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-3:] == second[:3]  # 3-token overlap
    assert chunks[-1].text.endswith("Word24")  # nothing lost at the end


def test_text_is_sliced_from_original_preserving_case():
    chunks = _chunks()
    for c in chunks:
        assert TEXT[c.start_char : c.end_char] == c.text
    assert chunks[0].text.startswith("Word0")


def test_ids_are_stable_and_sequential():
    a, b = _chunks(), _chunks()
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.chunk_id for c in a][:2] == ["doc:000", "doc:001"]
    assert a[0].content_hash == b[0].content_hash


def test_short_and_empty_docs():
    assert len(_chunks("just a few words")) == 1
    assert _chunks("   ") == []


def test_chunks_do_not_start_mid_word():
    text = "Zoraxin is cleared by Velkase enzyme activity in hepatic tissue today"
    for c in _chunks(text, tok=SubwordTokenizer(), max_tokens=8, overlap=3):
        assert c.start_char == 0 or text[c.start_char - 1] == " "
