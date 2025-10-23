"""Base class for embedding models."""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models.

    Args:
        model_uri: HuggingFace model identifier (e.g., 'org/model-name').
    """

    def __init__(self, model_uri: str) -> None:
        self.model_uri = model_uri

    @abstractmethod
    def download(self, output_path: Path) -> Path:
        """Download and prepare model, saving to output_path.

        Args:
            output_path: Path where the model zip should be saved.
        """
