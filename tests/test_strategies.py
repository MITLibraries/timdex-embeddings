import pytest

from embeddings.strategies.base import BaseStrategy
from embeddings.strategies.full_record import FullRecordStrategy
from embeddings.strategies.processor import create_embedding_inputs
from embeddings.strategies.registry import get_strategy_class


def test_full_record_strategy_extracts_text():
    timdex_record = {"timdex_record_id": "test-123", "title": ["Test Title"]}
    strategy = FullRecordStrategy()

    text = strategy.extract_text(timdex_record)

    assert text == """timdex_record_id: "test-123"\ntitle: ["Test Title"]\n"""
    assert strategy.STRATEGY_NAME == "full_record"


def test_create_embedding_inputs_yields_cartesian_product():
    # two records
    timdex_records = iter(
        [
            {
                "timdex_record_id": "id-1",
                "run_id": "run-1",
                "run_record_offset": 0,
                "transformed_record": b'{"title": ["Record 1"]}',
            },
            {
                "timdex_record_id": "id-2",
                "run_id": "run-1",
                "run_record_offset": 1,
                "transformed_record": b'{"title": ["Record 2"]}',
            },
        ]
    )

    # single strategy (for now)
    strategies = ["full_record"]

    embedding_inputs = list(create_embedding_inputs(timdex_records, strategies))

    assert len(embedding_inputs) == 2
    assert embedding_inputs[0].timdex_record_id == "id-1"
    assert embedding_inputs[0].embedding_strategy == "full_record"
    assert embedding_inputs[1].timdex_record_id == "id-2"
    assert embedding_inputs[1].embedding_strategy == "full_record"


def test_get_strategy_class_returns_correct_class():
    strategy_class = get_strategy_class("full_record")
    assert strategy_class is FullRecordStrategy


def test_get_strategy_class_raises_for_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown strategy"):
        get_strategy_class("nonexistent_strategy")


def test_subclass_without_strategy_name_raises_type_error():
    with pytest.raises(TypeError, match="must define 'STRATEGY_NAME' class attribute"):

        class InvalidStrategy(BaseStrategy):
            pass


def test_subclass_with_non_string_strategy_name_raises_type_error():
    with pytest.raises(
        TypeError, match="must override 'STRATEGY_NAME' with a valid string"
    ):

        class InvalidStrategy(BaseStrategy):
            STRATEGY_NAME = 123
