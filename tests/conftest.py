import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from embeddings.models.base import BaseEmbeddingModel


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

    def download(self, output_path: Path) -> Path:
        """Create a fake model zip file for testing."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_path, "w") as zf:
            zf.writestr("config.json", '{"model": "mock", "vocab_size": 30000}')
            zf.writestr("pytorch_model.bin", b"fake model weights")
            zf.writestr("tokenizer.json", '{"version": "1.0"}')
        return output_path


@pytest.fixture
def mock_model():
    """Fixture providing a MockEmbeddingModel instance."""
    return MockEmbeddingModel()
