import json
from collections.abc import Iterator

from embeddings.embedding import EmbeddingInput
from embeddings.strategies.registry import get_strategy_class


def create_embedding_inputs(
    timdex_records: Iterator[dict],
    strategies: list[str],
) -> Iterator[EmbeddingInput]:
    """Yield EmbeddingInput instances for all records x all strategies.

    Creates a cartesian product: each record is processed by each strategy,
    yielding one EmbeddingInput per combination.

    Args:
        timdex_records: Iterator of TIMDEX records.
            Expected keys: timdex_record_id, run_id, run_record_offset,
            transformed_record (bytes)
        strategies: List of strategy names to apply

    Yields:
        EmbeddingInput instances ready for embedding model

    Example:
        100 records x 3 strategies = 300 EmbeddingInput instances
    """
    for timdex_record in timdex_records:
        # decode and parse the TIMDEX JSON record
        transformed_record = json.loads(timdex_record["transformed_record"].decode())

        # apply all strategies to the record and yield
        for strategy_name in strategies:
            strategy_class = get_strategy_class(strategy_name)
            strategy_instance = strategy_class(
                timdex_record_id=timdex_record["timdex_record_id"],
                run_id=timdex_record["run_id"],
                run_record_offset=timdex_record["run_record_offset"],
                transformed_record=transformed_record,
            )
            yield strategy_instance.to_embedding_input()
