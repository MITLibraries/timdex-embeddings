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


def test_download_model_unknown_uri(caplog, runner):
    result = runner.invoke(
        main, ["download-model", "--model-uri", "unknown/model", "--output", "out.zip"]
    )
    assert result.exit_code != 0
    assert "Unknown model URI" in caplog.text


def test_download_model_not_implemented(caplog, runner):
    caplog.set_level("INFO")
    result = runner.invoke(
        main,
        [
            "download-model",
            "--model-uri",
            "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte",
            "--output",
            "out.zip",
        ],
    )
    assert (
        "Downloading model: opensearch-project/"
        "opensearch-neural-sparse-encoding-doc-v3-gte, saving to: out.zip."
    ) in caplog.text
    assert result.exit_code != 0
