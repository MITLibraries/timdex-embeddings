import functools
import json
import logging
import time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import click
import jsonlines
from timdex_dataset_api import TIMDEXDataset

from embeddings.config import configure_logger, configure_sentry
from embeddings.models.registry import get_model_class
from embeddings.strategies.processor import create_embedding_inputs
from embeddings.strategies.registry import STRATEGY_REGISTRY

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from embeddings.models.base import BaseEmbeddingModel


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


def model_required(f: Callable) -> Callable:
    """Middleware decorator for commands that require an embedding model.

    This decorator adds two CLI options:
    - "--model-uri": defaults to environment variable "TE_MODEL_URI"
    - "--model-path": defaults to environment variable "TE_MODEL_PATH"

    The decorator intercepts these parameters, uses the model URI to identify and
    instantiate the appropriate embedding model class with the provided model path,
    and stores the model instance in the Click context at ctx.obj["model"].

    Both model_uri and model_path parameters are consumed by the decorator and not
    passed to the decorated command function.
    """

    @click.option(
        "--model-uri",
        envvar="TE_MODEL_URI",
        required=True,
        help="HuggingFace model URI (e.g., 'org/model-name')",
    )
    @click.option(
        "--model-path",
        required=True,
        envvar="TE_MODEL_PATH",
        type=click.Path(path_type=Path),
        help=(
            "Path where the model will be downloaded to and loaded from, "
            "e.g. '/path/to/model'."
        ),
    )
    @functools.wraps(f)
    def wrapper(*args: tuple, **kwargs: dict[str, str | Path]) -> Callable:
        # pop "model_uri" and "model_path" from CLI args
        model_uri: str = str(kwargs.pop("model_uri"))
        model_path: str | Path = str(kwargs.pop("model_path"))

        # initialize embedding class
        model_class = get_model_class(str(model_uri))
        model: BaseEmbeddingModel = model_class(model_path)
        logger.info(
            f"Embedding class '{model.__class__.__name__}' "
            f"initialized from model URI '{model_uri}'."
        )

        # save embedding class instance to Context
        ctx: click.Context = args[0]  # type: ignore[assignment]
        ctx.obj["model"] = model

        return f(*args, **kwargs)

    return wrapper


@main.command()
def ping() -> None:
    """Emit 'pong' to debug logs and stdout."""
    logger.debug("pong")
    click.echo("pong")


@main.command()
@click.pass_context
@model_required
def download_model(
    ctx: click.Context,
) -> None:
    """Download a model from HuggingFace and save locally."""
    model: BaseEmbeddingModel = ctx.obj["model"]

    logger.info(f"Downloading model: {model.model_uri}")
    result_path = model.download()

    message = f"Model downloaded and saved to: {result_path}"
    logger.info(message)
    click.echo(result_path)


@main.command()
@click.pass_context
@model_required
def test_model_load(ctx: click.Context) -> None:
    """Test loading of embedding class and local model based on env vars.

    In a deployed context, the following env vars are expected:
        - TE_MODEL_URI
        - TE_MODEL_PATH

    With these set, the embedding class should be registered successfully and initialized,
    and the model loaded from a local copy.

    This CLI command is NOT used during normal workflows.  This is used primary
    during development and after model downloading/loading changes to ensure the model
    loads correctly.
    """
    model: BaseEmbeddingModel = ctx.obj["model"]
    model.load()
    click.echo("OK")


@main.command()
@click.pass_context
@model_required
@click.option(
    "-d",
    "--dataset-location",
    required=True,
    type=click.Path(),
    help="TIMDEX dataset location, e.g. 's3://timdex/dataset', to read records from.",
)
@click.option(
    "--run-id",
    required=True,
    type=str,
    help="TIMDEX ETL run id.",
)
@click.option(
    "--run-record-offset",
    required=True,
    type=int,
    default=0,
    help="TIMDEX ETL run record offset to start from, default = 0.",
)
@click.option(
    "--record-limit",
    required=True,
    type=int,
    default=None,
    help="Limit number of records after --run-record-offset, default = None (unlimited).",
)
@click.option(
    "--strategy",
    type=click.Choice(list(STRATEGY_REGISTRY.keys())),
    required=True,
    multiple=True,
    help=(
        "Pre-embedding record transformation strategy. "
        "Repeatable to apply multiple strategies."
    ),
)
@click.option(
    "--output-jsonl",
    required=False,
    type=str,
    default=None,
    help="Optionally write embeddings to local JSONLines file (primarily for testing).",
)
def create_embeddings(
    ctx: click.Context,
    dataset_location: str,
    run_id: str,
    run_record_offset: int,
    record_limit: int,
    strategy: list[str],
    output_jsonl: str,
) -> None:
    """Create embeddings for TIMDEX records."""
    model: BaseEmbeddingModel = ctx.obj["model"]
    model.load()

    # init TIMDEXDataset
    timdex_dataset = TIMDEXDataset(dataset_location)

    # query TIMDEX dataset for an iterator of records
    timdex_records = timdex_dataset.read_dicts_iter(
        columns=[
            "timdex_record_id",
            "run_id",
            "run_record_offset",
            "transformed_record",
        ],
        run_id=run_id,
        where=f"""run_record_offset >= {run_record_offset}""",
        limit=record_limit,
        action="index",
    )

    # create an iterator of EmbeddingInputs applying all requested strategies
    embedding_inputs = create_embedding_inputs(timdex_records, list(strategy))

    # create embeddings via the embedding model
    embeddings = model.create_embeddings(embedding_inputs)

    # if requested, write embeddings to a local JSONLines file
    if output_jsonl:
        with jsonlines.open(
            output_jsonl,
            mode="w",
            dumps=lambda obj: json.dumps(
                obj,
                default=str,
            ),
        ) as writer:
            for embedding in embeddings:
                writer.write(embedding.to_dict())

    # else, default writing embeddings back to TIMDEX dataset
    else:
        # WIP NOTE: write via anticipated timdex_dataset.embeddings.write(...)
        # NOTE: will likely use an imported TIMDEXEmbedding class from TDA, which the
        #   Embedding instance will nearly 1:1 map to.
        raise NotImplementedError

    logger.info("Embeddings creation complete.")


if __name__ == "__main__":  # pragma: no cover
    logger = logging.getLogger("embeddings.main")
    main()
