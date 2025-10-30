"""OpenSearch Neural Sparse Doc v3 GTE model."""

import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import snapshot_download
from transformers import AutoModelForMaskedLM, AutoTokenizer

from embeddings.embedding import Embedding, RecordText
from embeddings.models.base import BaseEmbeddingModel

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from transformers.models.distilbert.tokenization_distilbert_fast import (
        DistilBertTokenizerFast,
    )

logger = logging.getLogger(__name__)


class OSNeuralSparseDocV3GTE(BaseEmbeddingModel):
    """OpenSearch Neural Sparse Encoding Doc v3 GTE model.

    HuggingFace URI: opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte
    """

    MODEL_URI = "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte"

    def __init__(self, model_path: str | Path) -> None:
        """Initialize the model.

        Args:
            model_path: Path where the model will be downloaded to and loaded from.
        """
        super().__init__(model_path)
        self._model: PreTrainedModel | None = None
        self._tokenizer: DistilBertTokenizerFast | None = None
        self._special_token_ids: list | None = None
        self._id_to_token: list | None = None

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

        # ensure model exists locally
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at path: {self.model_path}")

        # load local model and tokenizer
        self._model = AutoModelForMaskedLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            self.model_path,
            local_files_only=True,
        )

        # setup special tokens
        self._special_token_ids = [
            self._tokenizer.vocab[str(token)]
            for token in self._tokenizer.special_tokens_map.values()
        ]

        # setup id_to_token mapping
        self._id_to_token = ["" for _ in range(self._tokenizer.vocab_size)]
        for token, token_id in self._tokenizer.vocab.items():
            self._id_to_token[token_id] = token

        logger.info(f"Model loaded successfully, {time.perf_counter()-start_time}s")

    def create_embedding(self, input_record: RecordText) -> Embedding:
        raise NotImplementedError
