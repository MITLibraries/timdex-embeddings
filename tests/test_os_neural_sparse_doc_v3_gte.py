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

    config_json = {
        "model_type": "distilbert",
        "vocab_size": 30000,
        "auto_map": {
            "AutoConfig": "Alibaba-NLP/new-impl--configuration.NewConfig",
            "AutoModel": "Alibaba-NLP/new-impl--modeling.NewModel",
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config_json))
    (model_dir / "modeling.py").write_text("# mock modeling code")
    (model_dir / "configuration.py").write_text("# mock configuration code")
    (model_dir / "tokenizer.json").write_text('{"version": "1.0"}')
    return model_dir


class DummySparseEncoder:
    """Lightweight stand-in for sentence-transformers SparseEncoder."""

    def __init__(self, document_vectors: list[torch.Tensor] | None = None) -> None:
        self.document_vectors = document_vectors or []

    def encode_document(self, texts, **kwargs):  # noqa: ARG002
        if isinstance(texts, list):
            if self.document_vectors:
                return self.document_vectors
            return [torch.tensor([float(i), float(i) + 0.1]) for i in range(len(texts))]
        return torch.tensor([0.1, 0.2])

    def decode(self, tensor):
        value = float(torch.sum(tensor).item())
        return [("sum", value)]

    def start_multi_process_pool(self, devices):
        return devices


def test_model_uri(tmp_path):
    """Test model_uri property returns correct HuggingFace URI."""
    model = OSNeuralSparseDocV3GTE(tmp_path / "model")
    assert (
        model.model_uri
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


def test_load_success(monkeypatch, os_neural_sparse_model_path):
    """Test successful load from local path assigns SparseEncoder."""
    sentinel = object()
    captured_kwargs = {}

    def fake_sparse_encoder(*args, **kwargs):
        captured_kwargs["args"] = args
        captured_kwargs["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(
        "embeddings.models.os_neural_sparse_doc_v3_gte.SparseEncoder",
        fake_sparse_encoder,
    )

    model = OSNeuralSparseDocV3GTE(os_neural_sparse_model_path)
    model.load()

    assert model._model is sentinel
    assert captured_kwargs["args"][0] == str(os_neural_sparse_model_path)
    assert captured_kwargs["kwargs"]["trust_remote_code"] is True


def test_load_file_not_found(tmp_path):
    """Test load raises FileNotFoundError for missing path."""
    path = tmp_path / "missing"
    model = OSNeuralSparseDocV3GTE(path)

    with pytest.raises(FileNotFoundError, match="Model not found at path"):
        model.load()


def test_create_embedding_returns_embedding_object(tmp_path):
    """Test create_embedding returns an Embedding object with correct attributes."""
    model = OSNeuralSparseDocV3GTE(tmp_path / "model")
    model._model = DummySparseEncoder()

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
    assert embedding.embedding_vector == pytest.approx([0.1, 0.2])
    assert embedding.embedding_token_weights == {"sum": pytest.approx(0.3)}


def test_create_embeddings_consumes_iterator_and_returns_embeddings(
    tmp_path, monkeypatch
):
    """Test create_embeddings yields Embeddings for generator inputs."""
    model = OSNeuralSparseDocV3GTE(tmp_path / "model")
    dummy_encoder = DummySparseEncoder(
        document_vectors=[
            torch.tensor([0.1, 0.2]),
            torch.tensor([0.3, 0.4]),
        ]
    )
    model._model = dummy_encoder
    monkeypatch.delenv("TE_NUM_WORKERS", raising=False)
    monkeypatch.delenv("TE_BATCH_SIZE", raising=False)

    embedding_inputs = [
        EmbeddingInput(
            timdex_record_id="id-1",
            run_id="run-1",
            run_record_offset=0,
            embedding_strategy="strategy-1",
            text="text 1",
        ),
        EmbeddingInput(
            timdex_record_id="id-2",
            run_id="run-1",
            run_record_offset=1,
            embedding_strategy="strategy-1",
            text="text 2",
        ),
    ]

    embeddings = list(model.create_embeddings(iter(embedding_inputs)))

    assert len(embeddings) == 2
    assert embeddings[0].timdex_record_id == "id-1"
    assert embeddings[0].embedding_vector == pytest.approx([0.1, 0.2])
    assert embeddings[1].timdex_record_id == "id-2"
    assert embeddings[1].embedding_vector == pytest.approx([0.3, 0.4])
