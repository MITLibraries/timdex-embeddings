import json

from embeddings.strategies.base import BaseStrategy


class FullRecordStrategy(BaseStrategy):
    """Serialize entire TIMDEX record JSON as embedding input."""

    STRATEGY_NAME = "full_record"

    def extract_text(self) -> str:
        """Serialize the entire transformed_record as JSON."""
        return json.dumps(self.transformed_record)
