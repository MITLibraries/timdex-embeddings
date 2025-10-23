import zipfile

import pytest

from embeddings.models.registry import MODEL_REGISTRY, get_model_class


def test_mock_model_instantiation(mock_model):
    assert mock_model.model_uri == "test/mock-model"


def test_mock_model_download_creates_zip(mock_model, tmp_path):
    output_path = tmp_path / "test_model.zip"
    result = mock_model.download(output_path)

    assert result == output_path
    assert output_path.exists()
    assert zipfile.is_zipfile(output_path)


def test_mock_model_download_contains_expected_files(mock_model, tmp_path):
    output_path = tmp_path / "test_model.zip"
    mock_model.download(output_path)

    with zipfile.ZipFile(output_path, "r") as zf:
        file_list = zf.namelist()
        assert "config.json" in file_list
        assert "pytorch_model.bin" in file_list
        assert "tokenizer.json" in file_list


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
