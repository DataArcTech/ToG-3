from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field
from .Embedding_Base import Embeddings

DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


class HuggingFaceEmbeddings(BaseModel, Embeddings):
    """HuggingFace sentence_transformers embedding models.

    This class wraps any model compatible with the `sentence-transformers` library
    (https://www.sbert.net/) and provides easy integration with downstream applications,
    such as retrieval or clustering tasks.

    To use, you should have the ``sentence_transformers`` python package installed.

    Supports both standard and prompt-based embedding models such as:
    - BAAI/bge-large-en
    - thenlper/gte-base / gte-large
    - hkunlp/instructor-xl
    - sentence-transformers/all-mpnet-base-v2 (non-prompt based)

    Args:
        model_name (str): Path or name of the HuggingFace model.
        model_kwargs (Dict[str, Any], optional): Keyword arguments to pass when loading the model.
            Common parameters include:
                - 'device': "cuda" / "cpu"
                - 'prompts': a dictionary mapping prompt names to prompt strings
                - 'default_prompt_name': default key to use from `prompts` if no prompt is specified during encoding
        encode_kwargs (Dict[str, Any], optional): Keyword arguments passed to the model's `encode()` method.
            Useful parameters include:
                - 'prompt_name': key of the prompt to use (must exist in `prompts`)
                - 'prompt': a raw string prompt (overrides `prompt_name`)
                - 'batch_size': encoding batch size
                - 'normalize_embeddings': whether to normalize the output embeddings

    Example:
        .. code-block:: python

            from langchain_huggingface import HuggingFaceEmbeddings

            model_name = "BAAI/bge-large-en"
            model_kwargs = {
                'device': 'cuda',
                'prompts': {
                    'query': 'Represent the question for retrieving supporting documents: ',
                    'passage': 'Represent the document for retrieval: '
                },
                'default_prompt_name': 'query'
            }
            encode_kwargs = {
                'normalize_embeddings': True,
                # optionally: 'prompt_name': 'passage'
            }

            hf = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )

            embeddings = hf.embed_documents(["What is the capital of France?"])
    """


    model_name: str = Field(description="Model name to use.", default=DEFAULT_MODEL_NAME)

    cache_folder: Optional[str] = Field(
        description="Cache folder for Hugging Face files.Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable.", default=None
    )

    model_kwargs: Dict[str, Any] = Field(description="Keyword arguments to pass when loading the model.", 
                                         default_factory=dict)
    encode_kwargs: Dict[str, Any] = Field(description="Keyword arguments to pass when calling the `encode` method of the Sentence Transformer model.", 
                                          default_factory=dict)
    multi_process: bool = Field(
        description="If True it will start a multi-process pool to process the encoding with several independent processes. Great for vast amount of texts.", 
        default=False
    )
    show_progress_bar: bool = Field(
        description="Whether to show a progress bar.", default=False
    )

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)
        try:
            import sentence_transformers  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            ) from exc

        self._client = sentence_transformers.SentenceTransformer(
            self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
        )

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        import sentence_transformers  # type: ignore[import]

        texts = list(map(lambda x: x.replace("\n", " "), texts))
        if self.multi_process:
            pool = self._client.start_multi_process_pool()
            embeddings = self._client.encode(texts, pool)
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
        else:
            embeddings = self._client.encode(
                texts,
                show_progress_bar=self.show_progress_bar,
                **self.encode_kwargs,  # type: ignore
            )

        if isinstance(embeddings, list):
            raise TypeError(
                "Expected embeddings to be a Tensor or a numpy array, "
                "got a list instead."
            )

        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]
