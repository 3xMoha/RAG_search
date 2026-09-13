import io
import termios
from unittest.mock import patch

import pytest

from rag.cli import main, print_results, read_query
from rag.search import SearchIndex


@pytest.mark.parametrize("key", ["\x03", "\x04", "\x1b", ""])
def test_exit_keys_restore_terminal(key):
    with (
        patch("rag.cli.sys.stdin") as stdin,
        patch("rag.cli.termios.tcgetattr", return_value=[123]),
        patch("rag.cli.tty.setraw"),
        patch("rag.cli.termios.tcsetattr") as restore,
    ):
        stdin.read.return_value = key
        assert read_query() is None
        restore.assert_called_once_with(stdin, termios.TCSADRAIN, [123])


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
