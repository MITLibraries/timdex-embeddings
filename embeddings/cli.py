import logging
import time
from datetime import timedelta

import click

from embeddings.config import configure_logger, configure_sentry

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


if __name__ == "__main__":  # pragma: no cover
    logger = logging.getLogger("embeddings.main")
    main()
