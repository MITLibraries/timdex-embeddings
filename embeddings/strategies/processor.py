import json
from collections.abc import Iterator

from embeddings.embedding import EmbeddingInput
from embeddings.strategies.registry import get_strategy_class


def create_embedding_inputs(
    timdex_dataset_records: Iterator[dict],
    strategies: list[str],
) -> Iterator[EmbeddingInput]:
    """Yield EmbeddingInput instances for all records x all strategies.

    Creates a cartesian product: each record is processed by each strategy,
    yielding one EmbeddingInput per combination.

    Args:
        timdex_dataset_records: Iterator of TIMDEX records.
            Expected keys: timdex_record_id, run_id, run_record_offset,
            transformed_record (bytes)
        strategies: List of strategy names to apply

    Yields:
        EmbeddingInput instances ready for embedding model

    Example:
        100 records x 3 strategies = 300 EmbeddingInput instances
    """
    # instantiate strategy transformers
    transformers = [get_strategy_class(strategy)() for strategy in strategies]

    # loop through records and apply all strategies, yielding an EmbeddingInput for each
    for timdex_dataset_record in timdex_dataset_records:

        # decode and parse the TIMDEX JSON record once for all requested strategies
        timdex_record = json.loads(timdex_dataset_record["transformed_record"].decode())

        for transformer in transformers:
            # prepare text for embedding from transformer strategy
            text = transformer.extract_text(timdex_record)

            # emit an EmbeddingInput instance
            yield EmbeddingInput(
                timdex_record_id=timdex_dataset_record["timdex_record_id"],
                run_id=timdex_dataset_record["run_id"],
                run_record_offset=timdex_dataset_record["run_record_offset"],
                embedding_strategy=transformer.STRATEGY_NAME,
                text=text,
            )
