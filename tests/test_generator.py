import io
import json
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import pytest

from rag.chunker import Chunk
from rag.generator import generate, request_answer


def test_empty_results_skip_model():
    with patch("rag.generator.request_answer") as request:
        assert generate("question", []) == "No relevant context found."
        request.assert_not_called()


def test_context_and_source_labels():
    results = [(Chunk("notes.md", 2, "RAG retrieves context."), 0.9)]
    with patch("rag.generator.request_answer", return_value="It retrieves context [1].") as request:
        answer = generate("What is RAG?", results, model="other", url="http://server")
    prompt, model, url = request.call_args.args
    assert "[1] notes.md (chunk 2)\nRAG retrieves context." in prompt
    assert "Question: What is RAG?" in prompt
    assert (model, url) == ("other", "http://server")
    assert "It retrieves context [1]." in answer
    assert "Retrieved sources:\n[1] notes.md (chunk 2)" in answer


def test_ollama_request():
    with patch("rag.generator.urlopen", return_value=io.BytesIO(b'{"response":" Answer "}')) as send:
        assert request_answer("question", "model", "http://localhost:11434/") == "Answer"
    request = send.call_args.args[0]
    payload = json.loads(request.data)
    assert request.full_url == "http://localhost:11434/api/generate"
    assert payload["model"] == "model"
    assert payload["prompt"] == "question"
    assert payload["stream"] is False
    assert payload["options"]["num_ctx"] == 4096


@pytest.mark.parametrize("error, message", [
    (URLError("refused"), "ollama serve"),
    (HTTPError("http://localhost", 404, "missing", {}, None), "ollama pull"),
])
def test_server_errors_are_actionable(error, message):
    with patch("rag.generator.urlopen", side_effect=error):
        with pytest.raises(ValueError, match=message):
            request_answer("question", "model", "http://localhost")


@pytest.mark.parametrize("body", [b'{}', b'{"response":""}'])
def test_invalid_answers(body):
    with patch("rag.generator.urlopen", return_value=io.BytesIO(body)):
        with pytest.raises(ValueError, match="Ollama returned"):
            request_answer("question", "model", "http://localhost")
