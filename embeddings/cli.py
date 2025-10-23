import logging
import time
from datetime import timedelta
from pathlib import Path

import click

from embeddings.config import configure_logger, configure_sentry
from embeddings.models.registry import get_model_class

logger = logging.getLogger(__name__)


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
@click.option(
    "--model-uri",
    required=True,
    help="HuggingFace model URI (e.g., 'org/model-name')",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(path_type=Path),
    help="Output path for zipped model (e.g., '/path/to/model.zip')",
)
def download_model(model_uri: str, output: Path) -> None:
    """Download a model from HuggingFace and save as zip file."""
    # load embedding model class
    model_class = get_model_class(model_uri)
    model = model_class(model_uri)

    # download model assets
    logger.info(f"Downloading model: {model_uri}")
    result_path = model.download(output)

    message = f"Model downloaded and saved to: {result_path}"
    logger.info(message)
    click.echo(result_path)


@main.command()
@click.option(
    "--model-uri",
    required=True,
    help="HuggingFace model URI (e.g., 'org/model-name')",
)
def create_embeddings(_model_uri: str) -> None:
    # TODO: docstring # noqa: FIX002
    raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover
    logger = logging.getLogger("embeddings.main")
    main()
