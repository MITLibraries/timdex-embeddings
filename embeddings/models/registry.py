"""Registry mapping model URIs to model classes."""

from embeddings.models.base import BaseEmbeddingModel
from embeddings.models.os_neural_sparse_doc_v3_gte import OSNeuralSparseDocV3GTE

MODEL_REGISTRY: dict[str, type[BaseEmbeddingModel]] = {
    "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte": (
        OSNeuralSparseDocV3GTE
    ),
}


def get_model_class(model_uri: str) -> type[BaseEmbeddingModel]:
    """Get model class for given URI.

    Args:
        model_uri: HuggingFace model identifier.

    Returns:
        Model class for the given URI.
    """
    if model_uri not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        msg = f"Unknown model URI: {model_uri}. Available models: {available}"
        raise ValueError(msg)
    return MODEL_REGISTRY[model_uri]
