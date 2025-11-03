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


@dataclass
class Embedding:
    """Encapsulates a single embedding.

    Args:
        (timdex_record_id, run_id, run_record_offset): composite key for TIMDEX record
        model_uri: model URI used to create the embedding
        embedding_strategy: strategy used to create text for embedding
        embedding: model embedding created from text
    """

    timdex_record_id: str
    run_id: str
    run_record_offset: int
    model_uri: str
    embedding_strategy: str
    embedding: dict | list[float]

    timestamp: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
    )

    def to_dict(self) -> dict:
        """Marshal to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), default=str)
