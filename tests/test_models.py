import zipfile

import pytest

from embeddings.embedding import RecordText
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
    input_record = RecordText(
        timdex_record_id="test-id",
        run_id="test-run",
        run_record_offset=42,
        embedding_strategy="full_record",
        text="test text",
    )
    embedding = mock_model.create_embedding(input_record)

    assert embedding.timdex_record_id == "test-id"
    assert embedding.run_id == "test-run"
    assert embedding.run_record_offset == 42
    assert embedding.embedding_strategy == "full_record"
    assert embedding.model_uri == "test/mock-model"
    assert embedding.embedding == {"coffee": 0.9, "seattle": 0.5}


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


def test_base_model_create_embeddings_calls_create_embedding(mock_model):
    input_records = [
        RecordText(
            timdex_record_id="id-1",
            run_id="run-1",
            run_record_offset=0,
            embedding_strategy="full_record",
            text="text 1",
        ),
        RecordText(
            timdex_record_id="id-2",
            run_id="run-1",
            run_record_offset=1,
            embedding_strategy="full_record",
            text="text 2",
        ),
    ]

    # create_embeddings should iterate and call create_embedding
    embeddings = list(mock_model.create_embeddings(iter(input_records)))

    assert len(embeddings) == 2  # two input records
    assert embeddings[0].timdex_record_id == "id-1"
    assert embeddings[1].timdex_record_id == "id-2"
