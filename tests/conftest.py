import json
import logging
import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from embeddings.embedding import Embedding, EmbeddingInput
from embeddings.models import registry
from embeddings.models.base import BaseEmbeddingModel

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _test_env(monkeypatch):
    monkeypatch.setenv("SENTRY_DSN", "None")
    monkeypatch.setenv("WORKSPACE", "test")


@pytest.fixture
def runner():
    return CliRunner()


class MockEmbeddingModel(BaseEmbeddingModel):
    """Simple test model that doesn't hit external APIs."""

    MODEL_URI = "test/mock-model"

    def __init__(self, model_path: str | Path) -> None:
        """Initialize the mock model."""
        super().__init__(model_path)

    def download(self) -> Path:
        """Create a fake model zip file for testing."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.model_path, "w") as zf:
            zf.writestr("config.json", '{"model": "mock", "vocab_size": 30000}')
            zf.writestr("pytorch_model.bin", b"fake model weights")
            zf.writestr("tokenizer.json", '{"version": "1.0"}')
        return self.model_path

    def load(self) -> None:
        logger.info("Model loaded successfully, 1.5s")

    def create_embedding(self, embedding_input: EmbeddingInput) -> Embedding:
        return Embedding(
            timdex_record_id=embedding_input.timdex_record_id,
            run_id=embedding_input.run_id,
            run_record_offset=embedding_input.run_record_offset,
            embedding_strategy=embedding_input.embedding_strategy,
            model_uri=self.model_uri,
            embedding_vector=[0.1, 0.2, 0.3],
            embedding_token_weights={"coffee": 0.9, "seattle": 0.5},
        )


@pytest.fixture
def mock_model(tmp_path):
    """Fixture providing a MockEmbeddingModel instance."""
    return MockEmbeddingModel(tmp_path / "model")


@pytest.fixture
def register_mock_model(monkeypatch):
    """Register MockEmbeddingModel in the model registry."""
    monkeypatch.setitem(registry.MODEL_REGISTRY, "test/mock-model", MockEmbeddingModel)
    monkeypatch.setenv("TE_MODEL_PATH", "/fake/path")


@pytest.fixture
def mock_snapshot_download(monkeypatch, tmp_path):
    """Mock snapshot_download to create fake model files locally."""

    def mock_snapshot(repo_id, local_dir, **kwargs):
        """Create fake model files based on repo_id."""
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)

        if repo_id == "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte":
            # create main model files
            config_json = {
                "model_type": "distilbert",
                "vocab_size": 30000,
                "auto_map": {
                    "AutoConfig": "Alibaba-NLP/new-impl--configuration.NewConfig",
                    "AutoModel": "Alibaba-NLP/new-impl--modeling.NewModel",
                },
            }
            (local_path / "config.json").write_text(json.dumps(config_json))
            (local_path / "pytorch_model.bin").write_bytes(b"fake weights")
            (local_path / "tokenizer.json").write_text('{"version": "1.0"}')

        elif repo_id == "Alibaba-NLP/new-impl":
            # create alibaba dependency files
            (local_path / "modeling.py").write_text("# Alibaba modeling code")
            (local_path / "configuration.py").write_text("# Alibaba configuration code")

        return str(local_path)

    monkeypatch.setattr(
        "embeddings.models.os_neural_sparse_doc_v3_gte.snapshot_download", mock_snapshot
    )
    return mock_snapshot
