import pytest

from graphrag.ingest.documents import load_documents, slugify


def test_loads_supported_files_sorted_with_slug_ids(tmp_path):
    (tmp_path / "b.md").write_text("Beta")
    (tmp_path / "A Doc.txt").write_text("Alpha")
    (tmp_path / "skip.pdf").write_text("x")
    (tmp_path / "empty.md").write_text("   ")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.md").write_text("Gamma")
    docs = load_documents(tmp_path)
    assert [d.doc_id for d in docs] == ["a-doc", "b", "sub-c"]


def test_collision_raises(tmp_path):
    (tmp_path / "x.md").write_text("1")
    (tmp_path / "x.txt").write_text("2")
    with pytest.raises(ValueError, match="collision"):
        load_documents(tmp_path)


def test_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nope")


def test_slugify():
    assert slugify("Halvern Therapeutics (v2)") == "halvern-therapeutics-v2"
