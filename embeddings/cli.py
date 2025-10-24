import functools
import logging
import time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import click

from embeddings.config import configure_logger, configure_sentry
from embeddings.models.registry import get_model_class

logger = logging.getLogger(__name__)


def model_required(f: Callable) -> Callable:
    """Decorator for commands that require a specific model."""

    @click.option(
        "--model-uri",
        envvar="TE_MODEL_URI",
        required=True,
        help="HuggingFace model URI (e.g., 'org/model-name')",
    )
    @functools.wraps(f)
    def wrapper(*args: list, **kwargs: dict) -> Callable:
        return f(*args, **kwargs)

    return wrapper


@click.group("embeddings")
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Pass to log at debug level instead of info",
)
@click.pass_context
def main(
    ctx: click.Context,
    *,
    verbose: bool,
) -> None:
    ctx.ensure_object(dict)
    ctx.obj["start_time"] = time.perf_counter()

    root_logger = logging.getLogger()
    logger.info(configure_logger(root_logger, verbose=verbose))
    logger.info(configure_sentry())
    logger.info("Running process")

    def _log_command_elapsed_time() -> None:
        elapsed_time = time.perf_counter() - ctx.obj["start_time"]
        logger.info(
            "Total time to complete process: %s", str(timedelta(seconds=elapsed_time))
        )

    ctx.call_on_close(_log_command_elapsed_time)


@main.command()
def ping() -> None:
    """Emit 'pong' to debug logs and stdout."""
    logger.debug("pong")
    click.echo("pong")


@main.command()
@model_required
@click.option(
    "--output",
    required=True,
    envvar="TE_MODEL_DOWNLOAD_PATH",
    type=click.Path(path_type=Path),
    help="Output path for zipped model (e.g., '/path/to/model.zip')",
)
def download_model(model_uri: str, output: Path) -> None:
    """Download a model from HuggingFace and save as zip file."""
    # load embedding model class
    model_class = get_model_class(model_uri)
    model = model_class()

    # download model assets
    logger.info(f"Downloading model: {model_uri}")
    result_path = model.download(output)

    message = f"Model downloaded and saved to: {result_path}"
    logger.info(message)
    click.echo(result_path)


@main.command()
@model_required
def create_embeddings(_model_uri: str) -> None:
    # TODO: docstring # noqa: FIX002
    raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover
    logger = logging.getLogger("embeddings.main")
    main()
