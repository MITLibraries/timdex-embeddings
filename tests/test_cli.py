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
        main,
        ["download-model", "--model-uri", "unknown/model", "--model-path", "out.zip"],
    )
    assert result.exit_code != 0
    assert "Unknown model URI" in caplog.text


def test_model_required_decorator_with_cli_option(
    caplog, register_mock_model, runner, tmp_path
):
    """Test decorator successfully initializes model from --model-uri option."""
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


def test_model_required_decorator_with_env_var(
    caplog, monkeypatch, register_mock_model, runner, tmp_path
):
    """Test decorator successfully initializes model from TE_MODEL_URI env var."""
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
    caplog, register_mock_model, runner, tmp_path
):
    """Test decorator stores model instance in ctx.obj['model']."""
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


def test_model_required_decorator_log_message(
    caplog, register_mock_model, runner, tmp_path
):
    """Test decorator logs correct initialization message."""
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


def test_model_required_decorator_works_across_commands(
    caplog, register_mock_model, runner
):
    """Test decorator works for multiple commands (test_model_load)."""
    result = runner.invoke(main, ["test-model-load", "--model-uri", "test/mock-model"])

    assert result.exit_code == 0
    assert (
        "Embedding class 'MockEmbeddingModel' initialized from model URI "
        "'test/mock-model'" in caplog.text
    )
    assert "OK" in result.output


def test_create_embeddings_requires_dataset_location(register_mock_model, runner):
    result = runner.invoke(main, ["create-embeddings", "--model-uri", "test/mock-model"])
    assert result.exit_code != 0
    assert "--dataset-location" in result.output


def test_create_embeddings_requires_run_id(register_mock_model, runner):
    result = runner.invoke(
        main,
        [
            "create-embeddings",
            "--model-uri",
            "test/mock-model",
            "--dataset-location",
            "s3://test",
        ],
    )
    assert result.exit_code != 0
    assert "Missing option '--run-id'" in result.output


def test_create_embeddings_requires_strategy(register_mock_model, runner):
    result = runner.invoke(
        main,
        [
            "create-embeddings",
            "--model-uri",
            "test/mock-model",
            "--dataset-location",
            "s3://test",
            "--run-id",
            "run-1",
        ],
    )
    assert result.exit_code != 0
    assert "Missing option '--strategy'" in result.output
