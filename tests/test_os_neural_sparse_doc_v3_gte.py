"""Tests for OSNeuralSparseDocV3GTE embedding model."""

# ruff: noqa: SLF001, PLR2004

import json
from pathlib import Path

import pytest

from embeddings.models.os_neural_sparse_doc_v3_gte import OSNeuralSparseDocV3GTE


def test_init(tmp_path):
    """Test model initialization."""
    model = OSNeuralSparseDocV3GTE(tmp_path / "model")
    assert model._model is None
    assert model._tokenizer is None
    assert model._special_token_ids is None
    assert model._id_to_token is None


def test_model_uri(tmp_path):
    """Test model_uri property returns correct HuggingFace URI."""
    model = OSNeuralSparseDocV3GTE(tmp_path / "model")
    assert (
        model.model_uri
        == "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte"
    )
    assert (
        model.MODEL_URI
        == "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte"
    )


def test_download_to_directory(
    neural_sparse_doc_v3_gte_mock_huggingface_snapshot, tmp_path
):
    """Test download to directory (not zip)."""
    output_path = tmp_path / "model_output"
    model = OSNeuralSparseDocV3GTE(output_path)

    result = model.download()

    assert result == output_path
    assert output_path.exists()
    assert (output_path / "config.json").exists()
    assert (output_path / "pytorch_model.bin").exists()
    assert (output_path / "tokenizer.json").exists()


def test_download_to_zip_file(
    neural_sparse_doc_v3_gte_mock_huggingface_snapshot, tmp_path
):
    """Test download creates zip when path ends in .zip."""
    output_path = tmp_path / "model.zip"
    model = OSNeuralSparseDocV3GTE(output_path)

    result = model.download()

    assert result == output_path
    assert output_path.exists()
    assert output_path.suffix == ".zip"


def test_download_calls_patch_method(
    neural_sparse_doc_v3_gte_mock_huggingface_snapshot, tmp_path, monkeypatch
):
    """Test that download calls the Alibaba patching method."""
    output_path = tmp_path / "model_output"
    model = OSNeuralSparseDocV3GTE(output_path)

    patch_called = False

    def mock_patch(temp_path):
        nonlocal patch_called
        patch_called = True

    monkeypatch.setattr(model, "_patch_local_model_with_alibaba_new_impl", mock_patch)

    model.download()

    assert patch_called


def test_download_returns_path(
    neural_sparse_doc_v3_gte_mock_huggingface_snapshot, tmp_path
):
    """Test download returns the output path."""
    output_path = tmp_path / "model_output"
    model = OSNeuralSparseDocV3GTE(output_path)

    result = model.download()

    assert result == output_path
    assert isinstance(result, Path)


def test_patch_downloads_alibaba_model(
    neural_sparse_doc_v3_gte_mock_huggingface_snapshot, tmp_path
):
    """Test patch method downloads Alibaba-NLP/new-impl."""
    model = OSNeuralSparseDocV3GTE(tmp_path / "model")
    model_temp_path = tmp_path / "temp_model"
    model_temp_path.mkdir()
    (model_temp_path / "config.json").write_text('{"model_type": "test"}')

    model._patch_local_model_with_alibaba_new_impl(model_temp_path)

    assert (model_temp_path / "modeling.py").exists()
    assert (model_temp_path / "configuration.py").exists()


def test_patch_copies_files(neural_sparse_doc_v3_gte_mock_huggingface_snapshot, tmp_path):
    """Test patch copies modeling.py and configuration.py."""
    model = OSNeuralSparseDocV3GTE(tmp_path / "model")
    model_temp_path = tmp_path / "temp_model"
    model_temp_path.mkdir()
    (model_temp_path / "config.json").write_text('{"model_type": "test"}')

    model._patch_local_model_with_alibaba_new_impl(model_temp_path)

    modeling_content = (model_temp_path / "modeling.py").read_text()
    config_content = (model_temp_path / "configuration.py").read_text()

    assert "Alibaba modeling code" in modeling_content
    assert "Alibaba configuration code" in config_content


def test_patch_updates_config_json(
    neural_sparse_doc_v3_gte_mock_huggingface_snapshot, tmp_path
):
    """Test patch updates auto_map in config.json."""
    model = OSNeuralSparseDocV3GTE(tmp_path / "model")
    model_temp_path = tmp_path / "temp_model"
    model_temp_path.mkdir()
    initial_config = {"model_type": "test", "vocab_size": 30000}
    (model_temp_path / "config.json").write_text(json.dumps(initial_config))

    model._patch_local_model_with_alibaba_new_impl(model_temp_path)

    updated_config = json.loads((model_temp_path / "config.json").read_text())

    assert "auto_map" in updated_config
    assert updated_config["auto_map"]["AutoConfig"] == "configuration.NewConfig"
    assert updated_config["auto_map"]["AutoModel"] == "modeling.NewModel"
    assert updated_config["auto_map"]["AutoModelForMaskedLM"] == "modeling.NewForMaskedLM"


def test_load_success(
    neural_sparse_doc_v3_gte_fake_model_directory,
    neural_sparse_doc_v3_gte_mock_transformers_models,
):
    """Test successful load from local path."""
    model = OSNeuralSparseDocV3GTE(neural_sparse_doc_v3_gte_fake_model_directory)

    model.load()

    assert model._model is not None
    assert model._tokenizer is not None


def test_load_file_not_found():
    """Test load raises FileNotFoundError for missing path."""
    nonexistent_path = Path("/nonexistent/path")
    model = OSNeuralSparseDocV3GTE(nonexistent_path)

    with pytest.raises(FileNotFoundError, match="Model not found at path"):
        model.load()


def test_load_initializes_model_and_tokenizer(
    neural_sparse_doc_v3_gte_fake_model_directory,
    neural_sparse_doc_v3_gte_mock_transformers_models,
):
    """Test load initializes _model and _tokenizer attributes."""
    model = OSNeuralSparseDocV3GTE(neural_sparse_doc_v3_gte_fake_model_directory)

    assert model._model is None
    assert model._tokenizer is None

    model.load()

    assert model._model is not None
    assert model._tokenizer is not None


def test_load_sets_up_special_token_ids(
    neural_sparse_doc_v3_gte_fake_model_directory,
    neural_sparse_doc_v3_gte_mock_transformers_models,
):
    """Test load sets up _special_token_ids list."""
    model = OSNeuralSparseDocV3GTE(neural_sparse_doc_v3_gte_fake_model_directory)

    model.load()

    assert model._special_token_ids is not None
    assert isinstance(model._special_token_ids, list)
    assert len(model._special_token_ids) == 3  # CLS, SEP, PAD
    assert 0 in model._special_token_ids  # [CLS] token id
    assert 1 in model._special_token_ids  # [SEP] token id
    assert 2 in model._special_token_ids  # [PAD] token id


def test_load_sets_up_id_to_token_mapping(
    neural_sparse_doc_v3_gte_fake_model_directory,
    neural_sparse_doc_v3_gte_mock_transformers_models,
):
    """Test load creates _id_to_token mapping correctly."""
    model = OSNeuralSparseDocV3GTE(neural_sparse_doc_v3_gte_fake_model_directory)

    model.load()

    assert model._id_to_token is not None
    assert isinstance(model._id_to_token, list)
    assert len(model._id_to_token) == 5  # vocab_size from mock
    assert model._id_to_token[0] == "[CLS]"
    assert model._id_to_token[1] == "[SEP]"
    assert model._id_to_token[2] == "[PAD]"
    assert model._id_to_token[3] == "word1"
    assert model._id_to_token[4] == "word2"
