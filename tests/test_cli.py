from embeddings.cli import main
from embeddings.models import registry
from tests.conftest import MockEmbeddingModel


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
        main,
        ["download-model", "--model-uri", "unknown/model", "--model-path", "out.zip"],
    )
    assert result.exit_code != 0
    assert "Unknown model URI" in caplog.text


def test_model_required_decorator_with_cli_option(caplog, monkeypatch, runner, tmp_path):
    """Test decorator successfully initializes model from --model-uri option."""
    monkeypatch.setitem(registry.MODEL_REGISTRY, "test/mock-model", MockEmbeddingModel)

    output_path = tmp_path / "model.zip"
    result = runner.invoke(
        main,
        [
            "download-model",
            "--model-uri",
            "test/mock-model",
            "--model-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert (
        "Embedding class 'MockEmbeddingModel' initialized from model URI "
        "'test/mock-model'" in caplog.text
    )
    assert output_path.exists()


def test_model_required_decorator_with_env_var(caplog, monkeypatch, runner, tmp_path):
    """Test decorator successfully initializes model from TE_MODEL_URI env var."""
    monkeypatch.setitem(registry.MODEL_REGISTRY, "test/mock-model", MockEmbeddingModel)
    monkeypatch.setenv("TE_MODEL_URI", "test/mock-model")

    output_path = tmp_path / "model.zip"
    result = runner.invoke(main, ["download-model", "--model-path", str(output_path)])

    assert result.exit_code == 0
    assert (
        "Embedding class 'MockEmbeddingModel' initialized from model URI "
        "'test/mock-model'" in caplog.text
    )
    assert output_path.exists()


def test_model_required_decorator_missing_parameter(runner):
    """Test decorator fails when --model-uri is not provided and env var is not set."""
    result = runner.invoke(main, ["download-model", "--model-path", "out.zip"])

    assert result.exit_code != 0
    assert "Missing option '--model-uri'" in result.output


def test_model_required_decorator_stores_model_in_context(
    caplog, monkeypatch, runner, tmp_path
):
    """Test decorator stores model instance in ctx.obj['model']."""
    monkeypatch.setitem(registry.MODEL_REGISTRY, "test/mock-model", MockEmbeddingModel)

    output_path = tmp_path / "model.zip"
    result = runner.invoke(
        main,
        [
            "download-model",
            "--model-uri",
            "test/mock-model",
            "--model-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    # verify the model was used successfully (download method was called)
    assert "Downloading model: test/mock-model" in caplog.text
    assert output_path.exists()


def test_model_required_decorator_log_message(caplog, monkeypatch, runner, tmp_path):
    """Test decorator logs correct initialization message."""
    monkeypatch.setitem(registry.MODEL_REGISTRY, "test/mock-model", MockEmbeddingModel)

    output_path = tmp_path / "model.zip"
    result = runner.invoke(
        main,
        [
            "download-model",
            "--model-uri",
            "test/mock-model",
            "--model-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert (
        "Embedding class 'MockEmbeddingModel' initialized from model URI "
        "'test/mock-model'" in caplog.text
    )


def test_model_required_decorator_works_across_commands(caplog, monkeypatch, runner):
    """Test decorator works for multiple commands (test_model_load)."""
    monkeypatch.setitem(registry.MODEL_REGISTRY, "test/mock-model", MockEmbeddingModel)
    monkeypatch.setenv("TE_MODEL_PATH", "/fake/path")

    result = runner.invoke(main, ["test-model-load", "--model-uri", "test/mock-model"])

    assert result.exit_code == 0
    assert (
        "Embedding class 'MockEmbeddingModel' initialized from model URI "
        "'test/mock-model'" in caplog.text
    )
    assert "OK" in result.output
