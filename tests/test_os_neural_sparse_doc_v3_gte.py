"""Tests for OSNeuralSparseDocV3GTE embedding model."""

# ruff: noqa: SLF001

import json
from pathlib import Path

import pytest
import torch

from embeddings.embedding import EmbeddingInput
from embeddings.models.os_neural_sparse_doc_v3_gte import OSNeuralSparseDocV3GTE


@pytest.fixture
def os_neural_sparse_model_path(tmp_path):
    """Create a fake model directory for OSNeuralSparseDocV3GTE."""
    model_dir = tmp_path / "fake_model"
    model_dir.mkdir()

    # create config.json
    config_json = {
        "model_type": "distilbert",
        "vocab_size": 30000,
        "auto_map": {
            "AutoConfig": "Alibaba-NLP/new-impl--configuration.NewConfig",
            "AutoModel": "Alibaba-NLP/new-impl--modeling.NewModel",
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config_json))

    # create modeling.py and configuration.py
    (model_dir / "modeling.py").write_text("# mock modeling code")
    (model_dir / "configuration.py").write_text("# mock configuration code")

    # create tokenizer files
    (model_dir / "tokenizer.json").write_text('{"version": "1.0"}')
    (model_dir / "vocab.txt").write_text("word1\nword2\n")

    return model_dir


@pytest.fixture
def mock_os_neural_sparse_transformers(monkeypatch):
    """Mock AutoTokenizer and AutoModelForMaskedLM for OSNeuralSparseDocV3GTE."""

    class MockTokenizer:
        """Mock tokenizer with necessary attributes."""

        def __init__(self, *args, **kwargs):  # noqa: ARG002
            self.vocab = {
                "[CLS]": 0,
                "[SEP]": 1,
                "[PAD]": 2,
                "coffee": 3,
                "mountain": 4,
                "seattle": 5,
            }
            self.vocab_size = len(self.vocab)
            self.special_tokens_map = {
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "pad_token": "[PAD]",
            }
            self.id_to_token = {v: k for k, v in self.vocab.items()}

        def convert_ids_to_tokens(self, token_ids):
            """Convert token IDs to token strings."""
            if isinstance(token_ids, int):
                return self.id_to_token.get(token_ids)
            return [self.id_to_token.get(tid) for tid in token_ids]

    class MockModel:
        """Mock model with necessary attributes."""

        def __init__(self, *args, **kwargs):  # noqa: ARG002
            self.config = {"vocab_size": 30000}

        def to(self, device):  # noqa: ARG002
            """Mock device placement."""
            return self

        def eval(self):
            """Mock evaluation mode."""
            return self

    class MockAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):  # noqa: ARG004
            return MockTokenizer()

    class MockAutoModelForMaskedLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):  # noqa: ARG004
            return MockModel()

    monkeypatch.setattr(
        "embeddings.models.os_neural_sparse_doc_v3_gte.AutoTokenizer",
        MockAutoTokenizer,
    )
    monkeypatch.setattr(
        "embeddings.models.os_neural_sparse_doc_v3_gte.AutoModelForMaskedLM",
        MockAutoModelForMaskedLM,
    )


@pytest.fixture
def loaded_model(os_neural_sparse_model_path, mock_os_neural_sparse_transformers):
    """Fixture providing a loaded OSNeuralSparseDocV3GTE instance."""
    model = OSNeuralSparseDocV3GTE(os_neural_sparse_model_path)
    model.load()
    return model


def test_init(tmp_path):
    """Test model initialization."""
    model = OSNeuralSparseDocV3GTE(tmp_path / "model")
    assert model._model is None
    assert model._tokenizer is None
    assert model._special_token_ids is None


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


def test_download_to_directory(mock_snapshot_download, tmp_path):
    """Test download to directory (not zip)."""
    output_path = tmp_path / "model_output"
    model = OSNeuralSparseDocV3GTE(output_path)

    result = model.download()

    assert result == output_path
    assert output_path.exists()
    assert (output_path / "config.json").exists()
    assert (output_path / "pytorch_model.bin").exists()
    assert (output_path / "tokenizer.json").exists()


def test_download_to_zip_file(mock_snapshot_download, tmp_path):
    """Test download creates zip when path ends in .zip."""
    output_path = tmp_path / "model.zip"
    model = OSNeuralSparseDocV3GTE(output_path)

    result = model.download()

    assert result == output_path
    assert output_path.exists()
    assert output_path.suffix == ".zip"


def test_download_calls_patch_method(mock_snapshot_download, tmp_path, monkeypatch):
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


def test_download_returns_path(mock_snapshot_download, tmp_path):
    """Test download returns the output path."""
    output_path = tmp_path / "model_output"
    model = OSNeuralSparseDocV3GTE(output_path)

    result = model.download()

    assert result == output_path
    assert isinstance(result, Path)


def test_patch_downloads_alibaba_model(mock_snapshot_download, tmp_path):
    """Test patch method downloads Alibaba-NLP/new-impl."""
    model = OSNeuralSparseDocV3GTE(tmp_path / "model")
    model_temp_path = tmp_path / "temp_model"
    model_temp_path.mkdir()
    (model_temp_path / "config.json").write_text('{"model_type": "test"}')

    model._patch_local_model_with_alibaba_new_impl(model_temp_path)

    assert (model_temp_path / "modeling.py").exists()
    assert (model_temp_path / "configuration.py").exists()


def test_patch_copies_files(mock_snapshot_download, tmp_path):
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


def test_patch_updates_config_json(mock_snapshot_download, tmp_path):
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
    os_neural_sparse_model_path,
    mock_os_neural_sparse_transformers,
):
    """Test successful load from local path."""
    model = OSNeuralSparseDocV3GTE(os_neural_sparse_model_path)

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
    os_neural_sparse_model_path,
    mock_os_neural_sparse_transformers,
):
    """Test load initializes _model and _tokenizer attributes."""
    model = OSNeuralSparseDocV3GTE(os_neural_sparse_model_path)

    assert model._model is None
    assert model._tokenizer is None

    model.load()

    assert model._model is not None
    assert model._tokenizer is not None


def test_load_sets_up_special_token_ids(
    os_neural_sparse_model_path,
    mock_os_neural_sparse_transformers,
):
    """Test load sets up _special_token_ids list."""
    model = OSNeuralSparseDocV3GTE(os_neural_sparse_model_path)

    model.load()

    assert model._special_token_ids is not None
    assert isinstance(model._special_token_ids, list)
    assert len(model._special_token_ids) == 3  # CLS, SEP, PAD
    assert 0 in model._special_token_ids  # [CLS] token id
    assert 1 in model._special_token_ids  # [SEP] token id
    assert 2 in model._special_token_ids  # [PAD] token id


def test_create_embedding_raises_error_if_model_not_loaded(tmp_path):
    """Test create_embedding raises RuntimeError if model not loaded."""
    model = OSNeuralSparseDocV3GTE(tmp_path / "model")
    embedding_input = EmbeddingInput(
        timdex_record_id="test:123",
        run_id="run-456",
        run_record_offset=0,
        embedding_strategy="title_only",
        text="test document",
    )

    with pytest.raises(RuntimeError, match="Model not loaded"):
        model.create_embedding(embedding_input)


def test_create_embedding_returns_embedding_object(tmp_path, monkeypatch):
    """Test create_embedding returns an Embedding object with correct attributes."""
    model = OSNeuralSparseDocV3GTE(tmp_path / "model")

    # mock _encode_documents
    def mock_encode_documents(texts):
        sparse_vector = torch.tensor([0.0, 0.0, 0.0, 0.91, 0.73])
        decoded_tokens = {"coffee": 0.91, "mountain": 0.73}
        return [(sparse_vector, decoded_tokens)]

    monkeypatch.setattr(model, "_encode_documents", mock_encode_documents)

    embedding_input = EmbeddingInput(
        timdex_record_id="test:123",
        run_id="run-456",
        run_record_offset=42,
        embedding_strategy="title_only",
        text="test document",
    )

    embedding = model.create_embedding(embedding_input)

    assert embedding.timdex_record_id == "test:123"
    assert embedding.run_id == "run-456"
    assert embedding.run_record_offset == 42
    assert embedding.model_uri == model.model_uri
    assert embedding.embedding_strategy == "title_only"
    assert isinstance(embedding.embedding_vector, list)
    assert embedding.embedding_vector == pytest.approx([0.0, 0.0, 0.0, 0.91, 0.73])
    assert embedding.embedding_token_weights == {"coffee": 0.91, "mountain": 0.73}


def test_decode_sparse_vectors_converts_to_token_weights(loaded_model):
    """Test _decode_sparse_vectors converts sparse vector to token-weight dict."""
    # sparse vector with weights for "coffee" (index 3) and "mountain" (index 4)
    sparse_vector = torch.tensor([0.0, 0.0, 0.0, 0.85, 0.62, 0.0])

    result = loaded_model._decode_sparse_vectors(sparse_vector)

    assert len(result) == 1
    assert result[0] == {"coffee": pytest.approx(0.85), "mountain": pytest.approx(0.62)}


def test_decode_sparse_vectors_with_empty_vector(loaded_model):
    """Test _decode_sparse_vectors returns empty dict for all-zero vector."""
    sparse_vector = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    result = loaded_model._decode_sparse_vectors(sparse_vector)

    assert len(result) == 1
    assert result[0] == {}
