import zipfile

import pytest

from embeddings.embedding import EmbeddingInput
from embeddings.models.base import BaseEmbeddingModel
from embeddings.models.registry import MODEL_REGISTRY, get_model_class


def test_mock_model_instantiation(mock_model):
    assert mock_model.model_uri == "test/mock-model"


def test_mock_model_download_creates_zip(mock_model):
    result = mock_model.download()

    assert result == mock_model.model_path
    assert mock_model.model_path.exists()
    assert zipfile.is_zipfile(mock_model.model_path)


def test_mock_model_download_contains_expected_files(mock_model):
    mock_model.download()

    with zipfile.ZipFile(mock_model.model_path, "r") as zf:
        file_list = zf.namelist()
        assert "config.json" in file_list
        assert "pytorch_model.bin" in file_list
        assert "tokenizer.json" in file_list


def test_mock_model_load(caplog, mock_model):
    mock_model.load()
    assert "Model loaded successfully" in caplog.text


def test_mock_model_create_embedding(mock_model):
    embedding_input = EmbeddingInput(
        timdex_record_id="test-id",
        run_id="test-run",
        run_record_offset=42,
        embedding_strategy="full_record",
        text="test text",
    )
    embedding = mock_model.create_embedding(embedding_input)

    assert embedding.timdex_record_id == "test-id"
    assert embedding.run_id == "test-run"
    assert embedding.run_record_offset == 42
    assert embedding.embedding_strategy == "full_record"
    assert embedding.model_uri == "test/mock-model"
    assert embedding.embedding_vector == [0.1, 0.2, 0.3]
    assert embedding.embedding_token_weights == {"coffee": 0.9, "seattle": 0.5}


def test_registry_contains_opensearch_model():
    assert (
        "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte"
        in MODEL_REGISTRY
    )


def test_get_model_class_returns_correct_class():
    model_class = get_model_class(
        "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte"
    )
    assert model_class.__name__ == "OSNeuralSparseDocV3GTE"


def test_get_model_class_raises_for_unknown_uri():
    with pytest.raises(ValueError, match="Unknown model URI"):
        get_model_class("unknown/model-uri")


def test_subclass_without_model_uri_raises_type_error():
    with pytest.raises(TypeError, match="must define 'MODEL_URI' class attribute"):

        class InvalidModel(BaseEmbeddingModel):
            pass


def test_subclass_with_non_string_model_uri_raises_type_error():
    with pytest.raises(TypeError, match="must override 'MODEL_URI' with a valid string"):

        class InvalidModel(BaseEmbeddingModel):
            MODEL_URI = 123
