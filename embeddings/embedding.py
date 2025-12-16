import datetime
import json
from dataclasses import asdict, dataclass, field


@dataclass
class EmbeddingInput:
    """Encapsulates the inputs for an embedding.

    When creating an embedding, we need to note what TIMDEX record the embedding is
    associated with and what strategy was used to prepare the embedding input text from
    the record itself.

    Args:
        (timdex_record_id, run_id, run_record_offset): composite key for TIMDEX record
        embedding_strategy: strategy used to create text for embedding
        text: text to embed, created from the TIMDEX record via the embedding_strategy
    """

    timdex_record_id: str
    run_id: str
    run_record_offset: int
    embedding_strategy: str
    text: str

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"<EmbeddingInput - record:'{self.timdex_record_id}', "
            f"strategy:'{self.embedding_strategy}', text length:{len(self.text)}>"
        )


@dataclass
class Embedding:
    """Encapsulates a single embedding.

    Args:
        (timdex_record_id, run_id, run_record_offset): composite key for TIMDEX record
        model_uri: model URI used to create the embedding
        embedding_strategy: strategy used to create text for embedding
        embedding_vector: vector representation of embedding
        embedding_token_weights: decoded token:weight pairs from sparse vector
            - only applicable to models that produce this output
    """

    timdex_record_id: str
    run_id: str
    run_record_offset: int
    model_uri: str
    embedding_strategy: str
    embedding_vector: list[float] | None
    embedding_token_weights: dict | None

    embedding_timestamp: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
    )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"<Embedding - record:'{self.timdex_record_id}', "
            f"strategy:'{self.embedding_strategy}'>"
        )

    def to_dict(self) -> dict:
        """Marshal to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), default=str)
