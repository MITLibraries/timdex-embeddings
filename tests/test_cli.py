import json
from pathlib import Path

import numpy as np
from timdex_dataset_api import TIMDEXDataset

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


def test_create_embeddings_writes_to_timdex_dataset(
    caplog,
    runner,
    dataset_with_records,
    register_mock_model,
):
    caplog.set_level("DEBUG")

    result = runner.invoke(
        main,
        [
            "--verbose",
            "create-embeddings",
            "--model-uri",
            "test/mock-model",
            "--dataset-location",
            dataset_with_records.location,
            "--run-id",
            "run-1",
            "--strategy",
            "full_record",
        ],
    )

    # assert CLI logged and exited cleanly
    assert result.exit_code == 0
    assert "total files: 1, total rows: 2" in caplog.text

    # reload temp TIMDEXDataset post embeddings write
    timdex_dataset = TIMDEXDataset(location=dataset_with_records.location)

    # assert embeddings written
    assert Path(timdex_dataset.embeddings.data_embeddings_root).exists()
    embeddings_df = timdex_dataset.embeddings.read_dataframe(run_id="run-1")
    assert len(embeddings_df) == 2

    # assert embedding row structure
    embedding_row = embeddings_df.iloc[0]
    assert embedding_row.embedding_model == "test/mock-model"
    assert embedding_row.embedding_strategy == "full_record"
    assert isinstance(json.loads(embedding_row.embedding_object), dict)
    assert isinstance(embedding_row.embedding_vector, np.ndarray)


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


def test_create_embeddings_requires_dataset_location(register_mock_model, runner):
    result = runner.invoke(
        main,
        [
            "create-embeddings",
            "--model-uri",
            "test/mock-model",
            "--run-id",
            "run-1",
            "--strategy",
            "full_record",
        ],
    )
    assert result.exit_code != 0
    assert "Both '--dataset-location' and '--run-id' are required" in result.output


def test_create_embeddings_requires_run_id(register_mock_model, runner):
    result = runner.invoke(
        main,
        [
            "create-embeddings",
            "--model-uri",
            "test/mock-model",
            "--dataset-location",
            "s3://test",
            "--strategy",
            "full_record",
        ],
    )
    assert result.exit_code != 0
    assert "Both '--dataset-location' and '--run-id' are required" in result.output


def test_create_embeddings_optional_input_jsonl(register_mock_model, runner, tmp_path):
    input_file = "tests/fixtures/cli_inputs/test-3-records.jsonl"
    output_file = tmp_path / "output.jsonl"

    result = runner.invoke(
        main,
        [
            "create-embeddings",
            "--model-uri",
            "test/mock-model",
            "--input-jsonl",
            input_file,
            "--strategy",
            "full_record",
            "--output-jsonl",
            str(output_file),
        ],
    )
    assert result.exit_code == 0
    assert output_file.exists()


def test_create_embeddings_optional_input_jsonl_does_not_require_dataset_params(
    register_mock_model, runner, tmp_path
):
    input_file = "tests/fixtures/cli_inputs/test-3-records.jsonl"
    output_file = tmp_path / "output.jsonl"

    result = runner.invoke(
        main,
        [
            "create-embeddings",
            "--model-uri",
            "test/mock-model",
            "--input-jsonl",
            input_file,
            "--strategy",
            "full_record",
            "--output-jsonl",
            str(output_file),
        ],
    )
    assert result.exit_code == 0
