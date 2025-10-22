from embeddings.cli import main


def test_cli_default_logging(caplog, runner):
    result = runner.invoke(main, ["ping"])
    assert result.exit_code == 0
    assert "Logger 'root' configured with level=INFO" in caplog.text


def test_cli_debug_logging(caplog, runner):
    with caplog.at_level("DEBUG"):
        result = runner.invoke(main, ["--verbose", "ping"])
    assert result.exit_code == 0
    assert "Logger 'root' configured with level=DEBUG" in caplog.text
    assert "pong" in caplog.text
    assert "pong" in result.output
