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
    try:
        model_class = get_model_class(model_uri)
    except ValueError as e:
        logger.exception("Unknown model URI: %s", model_uri)
        raise click.ClickException(str(e)) from e

    logger.info("Downloading model: %s", model_uri)
    model = model_class(model_uri)

    try:
        result_path = model.download(output)
        logger.info("Model downloaded successfully to: %s", result_path)
        click.echo(f"Model saved to: {result_path}")
    except NotImplementedError as e:
        logger.exception("Download not yet implemented for model: %s", model_uri)
        raise click.ClickException(str(e)) from e
    except Exception as e:
        logger.exception("Failed to download model: %s", model_uri)
        msg = f"Download failed: {e}"
        raise click.ClickException(msg) from e


@main.command()
@click.option(
    "--model-uri",
    required=True,
    help="HuggingFace model URI (e.g., 'org/model-name')",
)
def create_embeddings(model_uri: str) -> None:
    """Create embeddings."""
    logger.info("create-embeddings command called with model: %s", model_uri)
    raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover
    logger = logging.getLogger("embeddings.main")
    main()
