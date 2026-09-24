import io
from unittest.mock import Mock, patch

import pytest

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from rag.cli import main, print_results, read_query, prompt_session
from rag.search import SearchIndex


@pytest.mark.parametrize("interactive", [False, True])
def test_answer_mode(interactive, capsys):
    argv = ["rag-search", "--answer", "--model", "small"]
    if not interactive:
        argv.append("question")
    with (
        patch("sys.argv", argv),
        patch("rag.cli.SearchIndex") as index,
        patch("rag.cli.generate", return_value="Generated answer") as generate,
        patch("rag.cli.interpret", return_value="corrected question") as interpret,
        patch("rag.cli.warm_up") as warm_up,
        patch("rag.cli.read_query", side_effect=["question", None]) as read,
    ):
        events = Mock()
        for name, mock in [("warm", warm_up), ("read", read), ("interpret", interpret),
                           ("search", index.return_value.search), ("generate", generate)]:
            events.attach_mock(mock, name)
        index.return_value.search.return_value = []
        main()
    warm_up.assert_called_once_with(model="small", url="http://localhost:11434")
    interpret.assert_called_once_with("question", model="small", url="http://localhost:11434")
    index.return_value.search.assert_called_once_with("corrected question", top_k=1, min_score=0.25)
    names = [event[0] for event in events.mock_calls]
    assert names == (["warm", "read", "interpret", "search", "generate", "read"] if interactive
                     else ["warm", "interpret", "search", "generate"])
    generate.assert_called_once_with("question", [], model="small", url="http://localhost:11434")
    assert "Generated answer" in capsys.readouterr().out


@pytest.mark.parametrize("keys, expected", [
    ("\x03", None),
    ("\x1b", None),
    ("\x04hello\r", "hello"),
    ("ac\x1b[Db\r", "abc"),
    ("abc\x1b[D\x1b[DX\x1b[CY\r", "aXbYc"),
    ("abc\x1b[D\x7f\r", "ac"),
])
def test_prompt_keys(keys, expected):
    with create_pipe_input() as pipe:
        session = prompt_session(input=pipe, output=DummyOutput())
        pipe.send_text(keys)
        assert read_query(session) == expected


def test_prompt_history():
    with create_pipe_input() as pipe:
        session = prompt_session(input=pipe, output=DummyOutput())
        for question in ("first", "second"):
            pipe.send_text(question + "\r")
            assert read_query(session) == question
        pipe.send_text("\x1b[A\x1b[A\x1b[B\r")
        assert read_query(session) == "second"
        pipe.send_text("draft\x1b[A\x1b[B\r")
        assert read_query(session) == "draft"


def test_piped_input():
    with patch("rag.cli.sys.stdin", io.StringIO("hello\n")):
        assert read_query() == "hello"
        assert read_query() is None


def test_no_matches(capsys):
    print_results([])
    assert capsys.readouterr().out == "No matching results.\n"


@pytest.mark.parametrize("missing", [True, False])
def test_unusable_directory_skips_model(tmp_path, missing):
    directory = tmp_path / "missing" if missing else tmp_path
    with patch("rag.search.Embedder") as embedder:
        with pytest.raises(ValueError, match="data directory|No nonempty"):
            SearchIndex(directory)
        embedder.assert_not_called()


def test_missing_directory_cli(tmp_path, capsys):
    with patch("sys.argv", ["rag-search", "--data", str(tmp_path / "missing")]):
        with pytest.raises(SystemExit) as error:
            main()
    assert error.value.code == 1
    assert "Not a data directory" in capsys.readouterr().err


def test_cli_rebuild_uses_data_directory(tmp_path):
    with (
        patch("sys.argv", ["rag-search", "question", "--data", str(tmp_path), "--rebuild"]),
        patch("rag.cli.SearchIndex") as index,
        patch("rag.cli.warm_up") as warm_up,
        patch("rag.cli.interpret") as interpret,
    ):
        index.return_value.search.return_value = []
        main()
    warm_up.assert_not_called()
    interpret.assert_not_called()
    index.assert_called_once_with(
        tmp_path, max_chars=500, index_path=tmp_path / ".rag-index.json", rebuild=True,
    )
