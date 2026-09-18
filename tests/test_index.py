import json
from unittest.mock import Mock

import pytest

from rag.search import SearchIndex


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "notes.md").write_text("Original text", encoding="utf-8")
    embedder = Mock(model_name="test-model")
    embedder.embed_documents.return_value = [[1.0, 0.0]]
    embedder.embed_query.return_value = [1.0, 0.0]
    return tmp_path, tmp_path / ".rag-index.json", embedder


def test_reload_reuses_vectors_and_preserves_results(corpus):
    directory, path, embedder = corpus
    original = SearchIndex(directory, embedder, index_path=path)
    expected = original.search("question")
    embedder.embed_documents.reset_mock()
    (directory / "notes.md").unlink()

    loaded = SearchIndex(directory, embedder, index_path=path)

    assert loaded.search("question") == expected
    embedder.embed_documents.assert_not_called()


def test_rebuild_refreshes_documents(corpus):
    directory, path, embedder = corpus
    SearchIndex(directory, embedder, index_path=path)
    (directory / "notes.md").write_text("Updated text", encoding="utf-8")
    refreshed = SearchIndex(directory, embedder, index_path=path, rebuild=True)

    embedder.embed_documents.assert_called_with(["Updated text"])
    assert refreshed.chunks[0].text == "Updated text"
    assert SearchIndex(directory, embedder, index_path=path).chunks == refreshed.chunks


@pytest.mark.parametrize("changed", ["model", "max_chars", "directory"])
def test_changed_settings_require_rebuild(corpus, changed):
    directory, path, embedder = corpus
    SearchIndex(directory, embedder, index_path=path)
    max_chars = 500
    if changed == "model":
        embedder.model_name = "another-model"
    elif changed == "max_chars":
        max_chars = 100
    else:
        directory = directory / "other"
        directory.mkdir()
    with pytest.raises(ValueError, match="--rebuild"):
        SearchIndex(directory, embedder, max_chars, index_path=path)


def test_corrupt_index_can_be_rebuilt(corpus):
    directory, path, embedder = corpus
    path.write_text("broken JSON", encoding="utf-8")
    with pytest.raises(ValueError, match="--rebuild"):
        SearchIndex(directory, embedder, index_path=path)
    SearchIndex(directory, embedder, index_path=path, rebuild=True)
    assert json.loads(path.read_text())["chunks"][0]["text"] == "Original text"
