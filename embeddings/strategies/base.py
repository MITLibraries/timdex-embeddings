from abc import ABC, abstractmethod

from embeddings.embedding import EmbeddingInput


class BaseStrategy(ABC):
    """Base class for embedding input strategies.

    All child classes must set class level attribute STRATEGY_NAME.
    """

    STRATEGY_NAME: str  # type hint to document the requirement

    def __init__(
        self,
        timdex_record_id: str,
        run_id: str,
        run_record_offset: int,
        transformed_record: dict,
    ) -> None:
        """Initialize strategy with TIMDEX record metadata.

        Args:
            timdex_record_id: TIMDEX record ID
            run_id: TIMDEX ETL run ID
            run_record_offset: record offset within the run
            transformed_record: parsed TIMDEX record JSON
        """
        self.timdex_record_id = timdex_record_id
        self.run_id = run_id
        self.run_record_offset = run_record_offset
        self.transformed_record = transformed_record

    def __init_subclass__(cls, **kwargs: dict) -> None:  # noqa: D105
        super().__init_subclass__(**kwargs)

        # require class level STRATEGY_NAME to be set
        if not hasattr(cls, "STRATEGY_NAME"):
            msg = f"{cls.__name__} must define 'STRATEGY_NAME' class attribute"
            raise TypeError(msg)
        if not isinstance(cls.STRATEGY_NAME, str):
            msg = f"{cls.__name__} must override 'STRATEGY_NAME' with a valid string"
            raise TypeError(msg)

    @abstractmethod
    def extract_text(self) -> str:
        """Extract text to be embedded from transformed_record."""

    def to_embedding_input(self) -> EmbeddingInput:
        """Create EmbeddingInput instance with strategy-specific extracted text."""
        return EmbeddingInput(
            timdex_record_id=self.timdex_record_id,
            run_id=self.run_id,
            run_record_offset=self.run_record_offset,
            embedding_strategy=self.STRATEGY_NAME,
            text=self.extract_text(),
        )
