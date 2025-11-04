import logging

from embeddings.strategies.base import BaseStrategy
from embeddings.strategies.full_record import FullRecordStrategy

logger = logging.getLogger(__name__)

STRATEGY_CLASSES = [
    FullRecordStrategy,
]

STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    strategy.STRATEGY_NAME: strategy for strategy in STRATEGY_CLASSES
}


def get_strategy_class(strategy_name: str) -> type[BaseStrategy]:
    """Get strategy class by name.

    Args:
        strategy_name: Name of the strategy to retrieve
    """
    if strategy_name not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        msg = f"Unknown strategy: {strategy_name}. Available: {available}"
        logger.error(msg)
        raise ValueError(msg)

    return STRATEGY_REGISTRY[strategy_name]
