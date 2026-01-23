# timdex-embeddings

A CLI application for creating embeddings for TIMDEX.

## Development

- To preview a list of available Makefile commands: `make help`
- To install with dev dependencies: `make install`
- To update dependencies: `make update`
- To run unit tests: `make test`
- To lint the repo: `make lint`
- To run the app: `embeddings --help`
  - see below for more details about running the CLI with `.env` files and arguments

### Building Docker Images

This project is unusual in that we have **two, distinct** Docker files for building:

- `Dockerfile-cpu`
  - targets an `arm64` architecture for CPU-only inference
  - targets AWS Fargate ECS environment
  - also a good fit for running locally on `arm64` machines
- `Dockerfile-gpu`
  - targets an `amd64` architecture for CPU or GPU inference
  - targets AWS EC2 compute environment

Note the Docker image build commands in the `Makefile`, allowing for building a CPU image, a GPU image, or both.  

Also note that due to the size of the AWS Deep Learning Container (DLC) base images, these images can be quite large (~4gb for CPU, ~16gb for GPU).  For successful builds locally, you may need to increase the "Disk usage limit" in your local Docker environment; observed failures at 50gb, success at 96gb.

See the following ADR for more background: [01-parallel-builds-both-archs.md](docs/adrs/01-parallel-builds-both-archs.md).

## Environment Variables

### Required

```shell
SENTRY_DSN=### If set to a valid Sentry DSN, enables Sentry exception monitoring. This is not needed for local development.
WORKSPACE=### Set to `dev` for local development, this will be set to `stage` and `prod` in those environments by Terraform.
```

### Optional

```shell
TE_MODEL_URI=# HuggingFace model URI
TE_MODEL_PATH=# Path where the model will be downloaded to and loaded from
HF_HUB_DISABLE_PROGRESS_BARS=#boolean to use progress bars for HuggingFace model downloads; defaults to 'true' in deployed contexts

TE_TORCH_DEVICE=# defaults to 'cpu', but can be set to 'mps' for Apple Silicon, or theoretically 'cuda' for GPUs
TE_BATCH_SIZE=# batch size for each inference worker, defaults to 32
TE_NUM_WORKERS=# number of parallel model inference workers, defaults to 1
TE_CHUNK_SIZE=# number of batches each parallel worker grabs; no effect if TE_NUM_WORKERS=1
OMP_NUM_THREADS=# torch env var that sets thread usage during inference, default is not setting and using torch defaults
MKL_NUM_THREADS=# torch env var that sets thread usage during inference, default is not setting and using torch defaults

EMBEDDING_BATCH_SIZE=# controls batch size sent to model for embedding generation, primary memory management knob, defaults to 100
```

## Configuring an Embedding Model

This CLI application is designed to create embeddings for input texts.  To do this, a pre-trained model must be identified and configured for use.  

To this end, there is a base embedding class `BaseEmbeddingModel` that is designed to be extended and customized for a particular embedding model.

Once an embedding class has been created, the preferred approach is to set env vars `TE_MODEL_URI` and `TE_MODEL_PATH` directly in the `Dockerfile` to a) download a local snapshot of the model during image build, and b) set this model as the default for the CLI.

This allows invoking the CLI without specifying a model URI or local location, allowing this model to serve as the default, e.g.:

```shell
uv run --env-file .env embeddings test-model-load
```

## CLI Commands

For local development, all CLI commands should be invoked with the following format to pickup environment variables from `.env`:

```shell
uv run --env-file .env embeddings <COMMAND> <ARGS>
```

### `ping`
```text
Usage: embeddings ping [OPTIONS]

  Emit 'pong' to debug logs and stdout.
```

### `download-model`
```text
Usage: embeddings download-model [OPTIONS]

  Download a model from HuggingFace and save locally.

Options:
  --model-uri TEXT   HuggingFace model URI (e.g., 'org/model-name')
                     [required]
  --model-path PATH  Path where the model will be downloaded to and loaded
                     from, e.g. '/path/to/model'.  [required]
  --help             Show this message and exit.
```

### `test-model-load`
```text
Usage: embeddings test-model-load [OPTIONS]

  Test loading of embedding class and local model based on env vars.

  In a deployed context, the following env vars are expected:     -
  TE_MODEL_URI     - TE_MODEL_PATH

  With these set, the embedding class should be registered successfully and
  initialized, and the model loaded from a local copy.

  This CLI command is NOT used during normal workflows.  This is used primary
  during development and after model downloading/loading changes to ensure the
  model loads correctly.

Options:
  --model-uri TEXT   HuggingFace model URI (e.g., 'org/model-name')
                     [required]
  --model-path PATH  Path where the model will be downloaded to and loaded
                     from, e.g. '/path/to/model'.  [required]
  --help             Show this message and exit.
```

### `create-embeddings`
```text
Usage: embeddings create-embeddings [OPTIONS]

  Create embeddings for TIMDEX records.

Options:
  --model-uri TEXT             HuggingFace model URI (e.g., 'org/model-name')
                               [required]
  --model-path PATH            Path where the model will be downloaded to and
                               loaded from, e.g. '/path/to/model'.  [required]
  --dataset-location PATH      TIMDEX dataset location, e.g.
                               's3://timdex/dataset', to read records from.
  --run-id TEXT                TIMDEX ETL run id.
  --run-record-offset INTEGER  TIMDEX ETL run record offset to start from,
                               default = 0.
  --record-limit INTEGER       Limit number of records after --run-record-
                               offset, default = None (unlimited).
  --input-jsonl TEXT           Optional filepath to JSONLines file containing
                               TIMDEX records to create embeddings from.
  --strategy [full_record]     Pre-embedding record transformation strategy.
                               Repeatable to apply multiple strategies.
                               [required]
  --output-jsonl TEXT          Optionally write embeddings to local JSONLines
                               file (primarily for testing).
  --batch-size INTEGER         Number of embeddings to process per batch.
  --help                       Show this message and exit.
```
