"""Base class for embedding models."""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models.

    All child classes must set class level attribute MODEL_URI.
    """

    MODEL_URI: str  # Type hint to document the requirement

    def __init__(self, model_path: str | Path) -> None:
        """Initialize the embedding model with a model path.

        Args:
            model_path: Path where the model will be downloaded to and loaded from.
        """
        self.model_path = Path(model_path)

    def __init_subclass__(cls, **kwargs: dict) -> None:  # noqa: D105
        super().__init_subclass__(**kwargs)

        # require class level MODEL_URI to be set
        if not hasattr(cls, "MODEL_URI"):
            msg = f"{cls.__name__} must define 'MODEL_URI' class attribute"
            raise TypeError(msg)
        if not isinstance(cls.MODEL_URI, str):
            msg = f"{cls.__name__} must override 'MODEL_URI' with a valid string"
            raise TypeError(msg)

    @property
    def model_uri(self) -> str:
        return self.MODEL_URI

    @abstractmethod
    def download(self) -> Path:
        """Download and prepare model, saving to self.model_path.

        Returns:
            Path where the model was saved.
        """

    @abstractmethod
    def load(self) -> None:
        """Load model from self.model_path."""
