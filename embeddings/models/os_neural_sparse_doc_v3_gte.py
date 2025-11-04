"""OpenSearch Neural Sparse Doc v3 GTE model."""

import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForMaskedLM, AutoTokenizer

from embeddings.embedding import Embedding, EmbeddingInput
from embeddings.models.base import BaseEmbeddingModel

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from transformers.models.distilbert.tokenization_distilbert_fast import (
        DistilBertTokenizerFast,
    )

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
        self._model: PreTrainedModel | None = None
        self._tokenizer: DistilBertTokenizerFast | None = None
        self._special_token_ids: list[int] | None = None
        self._device: torch.device = torch.device("cpu")

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

        # setup device (use CUDA if available, otherwise CPU)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            self.model_path,
            local_files_only=True,
        )

        # load model as AutoModelForMaskedLM (required for sparse embeddings)
        self._model = AutoModelForMaskedLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self._model.to(self._device)  # type: ignore[arg-type]
        self._model.eval()

        # set special token IDs (following model card pattern)
        # these will be zeroed out in the sparse vectors
        self._special_token_ids = [
            self._tokenizer.vocab[token]  # type: ignore[index]
            for token in self._tokenizer.special_tokens_map.values()
        ]

        logger.info(
            f"Model loaded successfully on {self._device}, "
            f"{time.perf_counter() - start_time:.2f}s"
        )

    def create_embedding(self, input_record: EmbeddingInput) -> Embedding:
        """Create sparse embeddings for the input text (document encoding).

        This method generates sparse document embeddings.

        Process follows the model card exactly:
        1. Tokenize the document
        2. Pass through the masked language model to get logits
        3. Convert logits to sparse vector
        6. Return both raw sparse vector and decoded token-weight pairs

        Args:
            input_record: The input containing text to embed
        """
        # generate the sparse embeddings
        sparse_vector, decoded_tokens = self._encode_documents([input_record.text])[0]

        # coerce sparse vector tensor into list[float]
        sparse_vector_list = sparse_vector.cpu().numpy().tolist()

        return Embedding(
            timdex_record_id=input_record.timdex_record_id,
            run_id=input_record.run_id,
            run_record_offset=input_record.run_record_offset,
            model_uri=self.model_uri,
            embedding_strategy=input_record.embedding_strategy,
            embedding_vector=sparse_vector_list,
            embedding_token_weights=decoded_tokens,
        )

    def _encode_documents(
        self,
        texts: list[str],
    ) -> list[tuple[torch.Tensor, dict[str, float]]]:
        """Encode documents into sparse vectors and decoded token weights.

        This follows the pattern outlined on the HuggingFace model card for document
        encoding.

        This method will accommodate a list of text inputs, and return a list of
        embeddings, but the calling base method create_embeddings() is a singular input +
        output.  This method keeps the ability to handle multiple inputs + outputs, in the
        event we want something like a create_multiple_embeddings() method in the future.

        The following is a rough approximation of receiving logits back from the model
        and converting this to a sparse vector which can then be decoded to token:weights:

        ----------------------------------------------------------------------------------
        Imagine your vocabulary is just 5 words: ["cat", "dog", "bird", "fish", "tree"]
        Vocabulary indices:                      [  0,     1,      2,      3,       4]

        1. MODEL RETURNS LOGITS
        Let's say you input the text: "cat and dog"
        After tokenization, you have 3 tokens at 3 sequence positions
        The model outputs logits - a score for EVERY vocab word at EVERY position:

        logits = [
            # Position 0 (word "cat"):  scores for each vocab word at this position
            [9.2,  1.1,  0.3,  0.5,  0.2],  # "cat" gets high score (9.2)

            # Position 1 (word "and" - not in our toy vocab, but tokenized somehow):
            [2.1,  1.8,  0.4,  0.3,  0.9],  # moderate scores everywhere

            # Position 2 (word "dog"):
            [0.8,  8.7,  0.2,  0.4,  0.1],  # "dog" gets high score (8.7)
        ]
        Shape: (3 positions, 5 vocab words)


        2. PRODUCE SPARSE VECTORS FROM LOGITS
        We collapse the sequence positions by taking the MAX score for each vocab word:

        sparse_vector = [
            max(9.2, 2.1, 0.8),  # "cat": take max across all 3 positions = 9.2
            max(1.1, 1.8, 8.7),  # "dog": take max = 8.7
            max(0.3, 0.4, 0.2),  # "bird": take max = 0.4
            max(0.5, 0.3, 0.4),  # "fish": take max = 0.5
            max(0.2, 0.9, 0.1),  # "tree": take max = 0.9
        ]

        Apply transformations (ReLU, double-log) to make it sparser:
        sparse_vector = [5.1, 4.8, 0.0, 0.0, 0.0]  # smaller values become 0

        Final result:
        {"cat": 5.1, "dog": 4.8}  # Only the relevant words have non-zero weights
        ----------------------------------------------------------------------------------

        Args:
            texts: list of strings to create embeddings for
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() before create_embedding.")

        # tokenize the input texts
        features = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",  # returns PyTorch tensors instead of Python lists
            return_token_type_ids=False,
        )

        # move to CPU or GPU device, depending on what's available
        features = {k: v.to(self._device) for k, v in features.items()}

        # get model logits output
        with torch.no_grad():
            output = self._model(**features)[0]

        # generate sparse vectors from model logits
        sparse_vectors = self._get_sparse_vectors(features, output)

        # decode to token-weight dictionaries
        decoded = self._decode_sparse_vectors(sparse_vectors)

        # return list of tuple(vector, decoded token weights) embedding results
        return [(sparse_vectors[i], decoded[i]) for i in range(len(texts))]

    def _get_sparse_vectors(
        self, features: dict[str, torch.Tensor], output: torch.Tensor
    ) -> torch.Tensor:
        """Convert model logits output to sparse vectors.

        This follows the HuggingFace model card exactly: https://huggingface.co/
        opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte#usage-huggingface

        This implements the get_sparse_vector function from the model card:
            1. Max pooling with attention mask
            2. log(1 + log(1 + relu())) transformation
            3. Zero out special tokens

        Args:
            features: Tokenizer output with attention_mask
            output: Model logits of shape (batch_size, seq_len, vocab_size)

        Returns:
            Sparse vectors of shape (batch_size, vocab_size)
        """
        # max pooling with attention mask
        values, _ = torch.max(output * features["attention_mask"].unsqueeze(-1), dim=1)

        # apply the v3 model activation
        values = torch.log(1 + torch.log(1 + torch.relu(values)))

        # zero out special tokens
        values[:, self._special_token_ids] = 0

        return values

    def _decode_sparse_vectors(
        self, sparse_vectors: torch.Tensor
    ) -> list[dict[str, float]]:
        """Convert sparse vectors to token-weight dictionaries.

        Handles both single vectors and batches, returning a list of dictionaries mapping
        token strings to their weights.

        Args:
            sparse_vectors: Tensor of shape (batch_size, vocab_size) or (vocab_size,)

        Returns:
            List of dictionaries with token-weight pairs
        """
        if sparse_vectors.dim() == 1:
            sparse_vectors = sparse_vectors.unsqueeze(0)

        # move to CPU for processing
        sparse_vectors_cpu = sparse_vectors.cpu()

        results: list[dict] = []
        for vector in sparse_vectors_cpu:

            # find non-zero indices and values
            nonzero_indices = torch.nonzero(vector, as_tuple=False).squeeze(-1)

            if nonzero_indices.numel() == 0:
                results.append({})
                continue

            # get weights
            weights = vector[nonzero_indices].tolist()

            # convert indices to token strings
            token_ids = nonzero_indices.tolist()
            tokens = self._tokenizer.convert_ids_to_tokens(token_ids)  # type: ignore[union-attr]

            # create token:weight dictionary
            token_dict = {
                token: weight
                for token, weight in zip(tokens, weights, strict=True)
                if token is not None
            }
            results.append(token_dict)

        return results
