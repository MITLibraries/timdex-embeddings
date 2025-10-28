# timdex-embeddings

A CLI application for creating embeddings for TIMDEX.

## Development

- To preview a list of available Makefile commands: `make help`
- To install with dev dependencies: `make install`
- To update dependencies: `make update`
- To run unit tests: `make test`
- To lint the repo: `make lint`
- To run the app: `my-app --help` (Note the hyphen `-` vs underscore `_` that matches the `project.scripts` in `pyproject.toml`)

## Environment Variables

### Required

```shell
SENTRY_DSN=### If set to a valid Sentry DSN, enables Sentry exception monitoring. This is not needed for local development.
WORKSPACE=### Set to `dev` for local development, this will be set to `stage` and `prod` in those environments by Terraform.
```

### Optional

```shell
TE_MODEL_URI=# HuggingFace model URI
TE_MODEL_DOWNLOAD_PATH=# Download location for model
HF_HUB_DISABLE_PROGRESS_BARS=#boolean to use progress bars for HuggingFace model downloads; defaults to 'true' in deployed contexts
```

## Configuring an Embedding Model

This CLI application is designed to create embeddings for input texts.  To do this, a pre-trained model must be identified and configured for use.  

To this end, there is a base embedding class `BaseEmbeddingModel` that is designed to be extended and customized for a particular embedding model.

Once an embedding class has been created, the preferred approach is to set env vars `TE_MODEL_URI` and `TE_MODEL_DOWNLOAD_PATH` directly in the `Dockerfile` to a) download a local snapshot of the model during image build, and b) set this model as the default for the CLI.

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

  Download a model from HuggingFace and save as zip file.

Options:
  --model-uri TEXT  HuggingFace model URI (e.g., 'org/model-name')  [required]
  --output PATH     Output path for zipped model (e.g., '/path/to/model.zip')
                    [required]
  --help            Show this message and exit.
```

### `create-embeddings`
```text
TODO...
```


