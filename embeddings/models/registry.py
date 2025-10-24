"""Registry mapping model URIs to model classes."""

import logging

from embeddings.models.base import BaseEmbeddingModel
from embeddings.models.os_neural_sparse_doc_v3_gte import OSNeuralSparseDocV3GTE

logger = logging.getLogger(__name__)

MODEL_CLASSES = [OSNeuralSparseDocV3GTE]

MODEL_REGISTRY: dict[str, type[BaseEmbeddingModel]] = {
    model.MODEL_URI: model for model in MODEL_CLASSES
}


def get_model_class(model_uri: str) -> type[BaseEmbeddingModel]:
    """Get an embedding model class via the HuggingFace model URI.

    Args:
        model_uri: HuggingFace model identifier.
    """
    if model_uri not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        msg = f"Unknown model URI: {model_uri}. Available models: {available}"
        logger.error(msg)
        raise ValueError(msg)

    return MODEL_REGISTRY[model_uri]
