"""OpenSearch Neural Sparse Doc v3 GTE model."""

import logging
from pathlib import Path

from embeddings.models.base import BaseEmbeddingModel

logger = logging.getLogger(__name__)


class OSNeuralSparseDocV3GTE(BaseEmbeddingModel):
    """OpenSearch Neural Sparse Encoding Doc v3 GTE model.

    HuggingFace URI: opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte
    """

    MODEL_URI = "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte"

    def download(self, output_path: Path) -> Path:
        """Download and prepare model, saving to output_path.

        Args:
            output_path: Path where the model zip should be saved.
        """
        logger.info(f"Downloading model: { self.model_uri}, saving to: {output_path}.")
        raise NotImplementedError
