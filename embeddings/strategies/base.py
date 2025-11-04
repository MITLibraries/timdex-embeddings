from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """Base class for embedding input strategies.

    All child classes must set class level attribute STRATEGY_NAME.
    """

    STRATEGY_NAME: str  # type hint to document the requirement

    def __init_subclass__(cls, **kwargs: dict) -> None:  # noqa: D105
        super().__init_subclass__(**kwargs)

        # require class level STRATEGY_NAME to be set
        if not hasattr(cls, "STRATEGY_NAME"):
            raise TypeError(f"{cls.__name__} must define 'STRATEGY_NAME' class attribute")
        if not isinstance(cls.STRATEGY_NAME, str):
            raise TypeError(
                f"{cls.__name__} must override 'STRATEGY_NAME' with a valid string"
            )

    @abstractmethod
    def extract_text(self, timdex_record: dict) -> str:
        """Extract text to be embedded from transformed_record.

        Args:
            timdex_record: TIMDEX JSON record ("transformed_record" in TIMDEX dataset)
        """
