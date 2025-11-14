import json

from embeddings.strategies.base import BaseStrategy


class FullRecordStrategy(BaseStrategy):
    """Serialize entire TIMDEX record JSON as embedding input."""

    STRATEGY_NAME = "full_record"

    def extract_text(self, timdex_record: dict) -> str:
        """Serialize the entire TIMDEX record.

        The final string form is:
            <field>: <value as JSON><newline>
            <field>: <value as JSON><newline>
            ...
        """
        final_string = ""
        for k, v in timdex_record.items():
            final_string += f"{k}: {json.dumps(v)}\n"
        return final_string
