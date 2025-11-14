"""OpenSearch Neural Sparse Doc v3 GTE model."""

import json
import logging
import os
import shutil
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import cast

from huggingface_hub import snapshot_download
from sentence_transformers.sparse_encoder import SparseEncoder
from torch import Tensor

from embeddings.embedding import Embedding, EmbeddingInput
from embeddings.models.base import BaseEmbeddingModel

logger = logging.getLogger(__name__)


class OSNeuralSparseDocV3GTE(BaseEmbeddingModel):
    """OpenSearch Neural Sparse Encoding Doc v3 GTE model.

    This model generates sparse embeddings for documents by using a masked language
    model's logits to identify the most relevant tokens.

    HuggingFace URI: opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte
    """

    MODEL_URI = "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte"

    def __init__(self, model_path: str | Path) -> None:
        """Initialize the model.

        Args:
            model_path: Path where the model will be downloaded to and loaded from.
        """
        super().__init__(model_path)
        self.device = os.getenv("TE_TORCH_DEVICE", "cpu")
        self._model: SparseEncoder = None  # type: ignore[assignment]

    def download(self) -> Path:
        """Download and prepare model, saving to self.model_path.

        Returns:
            Path where the model was saved.
        """
        start_time = time.perf_counter()

        logger.info(f"Downloading model: {self.model_uri}, saving to: {self.model_path}.")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # download snapshot of HuggingFace model
            snapshot_download(repo_id=self.model_uri, local_dir=temp_path)
            logger.debug("Model download complete.")

            # patch local model with files from dependency model "Alibaba-NLP/new-impl"
            self._patch_local_model_with_alibaba_new_impl(temp_path)

            # compress model directory as a zip file
            if self.model_path.suffix.lower() == ".zip":
                logger.debug("Creating zip file of model contents.")
                shutil.make_archive(
                    str(self.model_path.with_suffix("")), "zip", temp_path
                )

            # copy to output directory without zipping
            else:
                logger.debug(f"Copying model contents to {self.model_path}")
                if self.model_path.exists():
                    shutil.rmtree(self.model_path)
                shutil.copytree(temp_path, self.model_path)

        logger.info(f"Model downloaded successfully, {time.perf_counter() - start_time}s")
        return self.model_path

    def _patch_local_model_with_alibaba_new_impl(self, model_temp_path: Path) -> None:
        """Patch downloaded model with required assets from Alibaba-NLP/new-impl.

        Our main model, opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte,
        has configurations that attempt dynamic downloading of another model for files.
        This can be seen here: https://huggingface.co/opensearch-project/opensearch-
        neural-sparse-encoding-doc-v3-gte/blob/main/config.json#L6-L14.

        To avoid our deployed CLI application making requests to the HuggingFace API to
        retrieve these required files, which is problematic during high concurrency, we
        manually download these files and patch the model during our local download and
        save.

        This allows us to load the primary model without any HuggingFace API calls.
        """
        logger.info("Downloading custom code from Alibaba-NLP/new-impl")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            snapshot_download(
                repo_id="Alibaba-NLP/new-impl",
                local_dir=str(temp_path),
            )

            logger.info("Copying Alibaba code and updating config.json")
            shutil.copy(temp_path / "modeling.py", model_temp_path / "modeling.py")
            shutil.copy(
                temp_path / "configuration.py",
                model_temp_path / "configuration.py",
            )

            with open(model_temp_path / "config.json") as f:
                config_json = json.load(f)
                config_json["auto_map"] = {
                    "AutoConfig": "configuration.NewConfig",
                    "AutoModel": "modeling.NewModel",
                    "AutoModelForMaskedLM": "modeling.NewForMaskedLM",
                    "AutoModelForMultipleChoice": "modeling.NewForMultipleChoice",
                    "AutoModelForQuestionAnswering": "modeling.NewForQuestionAnswering",
                    "AutoModelForSequenceClassification": (
                        "modeling.NewForSequenceClassification"
                    ),
                    "AutoModelForTokenClassification": (
                        "modeling.NewForTokenClassification"
                    ),
                }
            with open(model_temp_path / "config.json", "w") as f:
                f.write(json.dumps(config_json))

            logger.debug("Dependency model Alibaba-NLP/new-impl downloaded and used.")

    def load(self) -> None:
        """Load the model from self.model_path."""
        start_time = time.perf_counter()
        logger.info(f"Loading model from: {self.model_path}")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at path: {self.model_path}")

        # load model as SparseEncoder
        self._model = SparseEncoder(
            str(self.model_path),
            trust_remote_code=True,
            model_kwargs={},
            device="cpu",
        )

        logger.info(f"Model loaded successfully, {time.perf_counter() - start_time:.2f}s")

    def create_embedding(self, embedding_input: EmbeddingInput) -> Embedding:
        """Create an Embedding for an EmbeddingInput.

        This model is configured to return a sparse vector of vocabulary token indices
        and weights, and a dictionary of decoded tokens and weights that had a weight
        > 0 in the sparse vector.

        Args:
            embedding_input: EmbeddingInput instance
        """
        sparse_vector = self._model.encode_document(embedding_input.text)
        sparse_vector = cast("Tensor", sparse_vector)
        return self._get_embedding_from_sparse_vector(embedding_input, sparse_vector)

    def create_embeddings(
        self,
        embedding_inputs: Iterator[EmbeddingInput],
    ) -> Iterator[Embedding]:
        """Yield Embeddings for multiple EmbeddingInputs.

        If env var TE_NUM_WORKERS is set and >1, the encoding lib sentence-transformers
        will automatically create a pool of worker processes to work in parallel.

        Note: currently 2+ workers in amd64 and arm64 Docker contexts immediately exits
        due to a "Bus Error".  It is recommended to omit the env var TE_NUM_WORKERS, or
        set to "1", in Docker contexts.

        Currently, we also fully consume the input EmbeddingInputs before we start
        embedding work.  This may change in future iterations if we move to batching
        embedding creation, so until then it's assumed that inputs to this method are
        memory safe for the full run.

        Args:
            embedding_inputs: iterator of EmbeddingInputs
        """
        # consume input EmbeddingInputs
        embedding_inputs_list = list(embedding_inputs)
        if not embedding_inputs_list:
            return

        # extract texts from all inputs
        texts = [embedding_input.text for embedding_input in embedding_inputs_list]

        # read env vars for configurations
        num_workers = int(os.getenv("TE_NUM_WORKERS", "1"))
        batch_size = int(os.getenv("TE_BATCH_SIZE", "32"))
        chunk_size_env = os.getenv("TE_CHUNK_SIZE")
        chunk_size = int(chunk_size_env) if chunk_size_env else None

        # configure for inference
        if num_workers > 1 or self.device == "mps":
            device = None
            pool = self._model.start_multi_process_pool(
                [self.device for _ in range(num_workers)]
            )
        else:
            device = self.device
            pool = None
        logger.info(
            f"Num workers: {num_workers}, batch size: {batch_size}, "
            f"chunk size: {chunk_size, }device: {device}, pool: {pool}"
        )

        # get sparse vector embedding for input text(s)
        inference_start = time.perf_counter()
        sparse_vectors = self._model.encode_document(
            texts,
            batch_size=batch_size,
            device=device,
            pool=pool,
            save_to_cpu=True,
            chunk_size=chunk_size,
        )
        logger.info(f"Inference elapsed: {time.perf_counter()-inference_start}s")
        sparse_vectors = cast("list[Tensor]", sparse_vectors)

        for i, embedding_input in enumerate(embedding_inputs_list):
            sparse_vector = sparse_vectors[i]
            sparse_vector = cast("Tensor", sparse_vector)
            yield self._get_embedding_from_sparse_vector(embedding_input, sparse_vector)

    def _get_embedding_from_sparse_vector(
        self,
        embedding_input: EmbeddingInput,
        sparse_vector: Tensor,
    ) -> Embedding:
        """Prepare Embedding from EmbeddingInput and calculated sparse vector.

        This shared method is used by create_embedding() and create_embeddings() to
        prepare and return an Embedding.  A sparse vector is provided, which is decoded
        into a dictionary of tokens:weights, and a final Embedding instance is returned.

        Args:
            embedding_input: EmbeddingInput
            sparse_vector: sparse vector returned by model
        """
        # get decoded dictionary of tokens:weights
        decoded_token_weights = self._model.decode(sparse_vector)
        decoded_token_weights = cast("list[tuple[str, float]]", decoded_token_weights)
        embedding_token_weights = dict(decoded_token_weights)

        # prepare sparse vector for JSON serialization
        embedding_vector = sparse_vector.to_dense().tolist()

        return Embedding(
            timdex_record_id=embedding_input.timdex_record_id,
            run_id=embedding_input.run_id,
            run_record_offset=embedding_input.run_record_offset,
            model_uri=self.model_uri,
            embedding_strategy=embedding_input.embedding_strategy,
            embedding_vector=embedding_vector,
            embedding_token_weights=embedding_token_weights,
        )
