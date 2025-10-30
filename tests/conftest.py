import json
import logging
import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from embeddings.embedding import Embedding, RecordText
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

    def create_embedding(self, input_record: RecordText) -> Embedding:
        return Embedding(
            timdex_record_id=input_record.timdex_record_id,
            run_id=input_record.run_id,
            run_record_offset=input_record.run_record_offset,
            embedding_strategy=input_record.embedding_strategy,
            model_uri=self.model_uri,
            embedding={"coffee": 0.9, "seattle": 0.5},
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
def neural_sparse_doc_v3_gte_fake_model_directory(tmp_path):
    """Create a fake downloaded model directory with required files."""
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
def neural_sparse_doc_v3_gte_mock_huggingface_snapshot(monkeypatch, tmp_path):
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


@pytest.fixture
def neural_sparse_doc_v3_gte_mock_transformers_models(monkeypatch):
    """Mock AutoModelForMaskedLM and AutoTokenizer."""

    class MockTokenizer:
        """Mock tokenizer with necessary attributes."""

        def __init__(self, *args, **kwargs):  # noqa: ARG002
            self.vocab = {
                "[CLS]": 0,
                "[SEP]": 1,
                "[PAD]": 2,
                "word1": 3,
                "word2": 4,
            }
            self.vocab_size = len(self.vocab)
            self.special_tokens_map = {
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "pad_token": "[PAD]",
            }

    class MockModel:
        """Mock model with necessary attributes."""

        def __init__(self, *args, **kwargs):  # noqa: ARG002
            self.config = {"vocab_size": 30000}

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

    return {"tokenizer": MockTokenizer, "model": MockModel}
