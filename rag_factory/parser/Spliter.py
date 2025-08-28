import re
import numpy as np
from rag_factory.Embed.Embedding_Base import Embeddings
from typing import (
    AbstractSet,
    Callable,
    Collection,
    List,
    Literal,
    Optional,
    Union,
    Dict,
    Tuple,
    cast
)

class MarkdownHeaderTextSplitter:
    """Splitting markdown files based on specified headers."""

    def __init__(
        self,
        headers_to_split_on: List[str] = ["#", "##"], 
        strip_headers: bool = True,
        chunk_size: int = 0
    ):
        """
        Args:
            headers_to_split_on: markdown标题开头格式
            strip_headers: 文本是否保留标题
            chunk_size: 最大chunk大小
        Return:{
            "content":
            "Header":{
                "level":
                "name":
            }
        }
        """
        self.headers_to_split_on = sorted(headers_to_split_on, key=lambda x: len(x), reverse=True)
        self.strip_headers = strip_headers
        self.chunk_size = chunk_size

    def _chunk_content(self, content: str) -> List[str]:
        """Split content into chunks of size `chunk_size` if needed."""
        if self.chunk_size <= 0:
            return [content]
        return [content[i:i+self.chunk_size] for i in range(0, len(content), self.chunk_size)]

    def split_text(self, text: str) -> List[Dict]:
        lines = text.split("\n")
        results = []

        current_content = []
        current_header = {"level": 0, "name": ""}
        header_stack = []
        in_code_block = False
        opening_fence = ""

        for line in lines:
            stripped_line = line.strip()
            stripped_line = "".join(filter(str.isprintable, stripped_line))

            # --- 代码块检测 ---
            if not in_code_block:
                if stripped_line.startswith("```") and stripped_line.count("```") == 1:
                    in_code_block = True
                    opening_fence = "```"
                elif stripped_line.startswith("~~~"):
                    in_code_block = True
                    opening_fence = "~~~"
            else:
                if stripped_line.startswith(opening_fence):
                    in_code_block = False
                    opening_fence = ""
            
            if in_code_block:
                current_content.append(line)
                continue

            matched_header = None
            for sep in self.headers_to_split_on:
                if stripped_line.startswith(sep) and (
                    len(stripped_line) == len(sep) or stripped_line[len(sep)] == " "
                ):
                    matched_header = sep
                    break

            if matched_header:
                # 先保留当前的文本
                if current_content:
                    section_text = "\n".join(current_content).strip()
                    for chunk in self._chunk_content(section_text):
                        results.append({
                            "content": chunk,
                            "Header": current_header.copy()
                        })
                    current_content = []

                # 更新 stack（去掉同级或更低级的 header）
                current_level = matched_header.count("#")
                while header_stack and header_stack[-1]["level"] >= current_level:
                    header_stack.pop()

                header_name = stripped_line[len(matched_header):].strip()
                current_header = {"level": current_level, "name": header_name}
                header_stack.append(current_header)

                if not self.strip_headers:
                    current_content.append(stripped_line + "\n")
            else:
                current_content.append(line)

        if current_content:
            section_text = "\n".join(current_content).strip()
            for chunk in self._chunk_content(section_text):
                results.append({
                    "content": chunk,
                    "Header": current_header.copy()
                })

        return results
        


from dataclasses import dataclass
@dataclass(frozen=True)
class Tokenizer:
    """Tokenizer data class."""

    chunk_overlap: int
    """Overlap in tokens between chunks"""
    tokens_per_chunk: int
    """Maximum number of tokens per chunk"""
    decode: Callable[[List[int]], str]
    """ Function to decode a list of token ids to a string"""
    encode: Callable[[str], List[int]]
    """ Function to encode a string to a list of token ids"""

def split_text_on_tokens(*, text: str, tokenizer: Tokenizer) -> List[str]:
    """Split incoming text and return chunks using tokenizer."""
    splits: List[str] = []
    input_ids = tokenizer.encode(text)
    start_idx = 0
    cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]
    while start_idx < len(input_ids):
        splits.append(tokenizer.decode(chunk_ids))
        if cur_idx == len(input_ids):
            break
        start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
        cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
    return splits


class TokenTextSplitter:
    """Splitting text to tokens using model tokenizer."""

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        encoding_name: str = "gpt2",
        model_name: Optional[str] = None,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
    ) -> None:
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed in order to for TokenTextSplitter. "
                "Please install it with `pip install tiktoken`."
            )

        if model_name is not None:
            enc = tiktoken.encoding_for_model(model_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._tokenizer = enc
        self._allowed_special = allowed_special
        self._disallowed_special = disallowed_special

    def split_text(self, text: str) -> List[str]:

        def _encode(_text: str) -> List[int]:
            return self._tokenizer.encode(
                _text,
                allowed_special=self._allowed_special,
                disallowed_special=self._disallowed_special,
            )

        tokenizer = Tokenizer(
            chunk_overlap=self._chunk_overlap,
            tokens_per_chunk=self._chunk_size,
            decode=self._tokenizer.decode,
            encode=_encode,
        )

        return split_text_on_tokens(text=text, tokenizer=tokenizer)


class RecursiveCharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 200,
        chunk_overlap: int = 0,
        separators: Optional[list[str]] = None,
        keep_separator: Union[bool, Literal["start", "end"]] = True,
        is_separator_regex: bool = False,
    ) -> None:
        
        if chunk_size <= 0:
            msg = f"chunk_size must be > 0, got {chunk_size}"
            raise ValueError(msg)
        if chunk_overlap < 0:
            msg = f"chunk_overlap must be >= 0, got {chunk_overlap}"
            raise ValueError(msg)
        if chunk_overlap > chunk_size:
            msg = (
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
            raise ValueError(msg)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap 
        self._separators = separators or ["\n\n", "\n", "#"]
        self._keep_separator = keep_separator
        self._is_separator_regex = is_separator_regex

    def split_text(self, text: str) -> list[str]:
        return self._split_recursive(text, self._separators)

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:

        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            return self._chunk_text(text)

        sep = separators[0]
        if self._is_separator_regex:
            parts = re.split(f"({sep})", text)  
        else:
            parts = text.split(sep)

        if len(parts) == 1:  
            return self._split_recursive(text, separators[1:])

        chunks, current = [], ""
        for i, p in enumerate(parts):
            if not p:
                continue
            if self._is_separator_regex:
                is_sep = bool(re.fullmatch(sep, p))
            else:
                is_sep = (p == sep)

            if is_sep:
                if self._keep_separator is True or self._keep_separator == "end":
                    current += p
                elif self._keep_separator == "start":
                    if current:
                        chunks.append(current)
                    current = p
                continue

            if len(current) + len(p) > self.chunk_size and current:
                chunks.extend(self._split_recursive(current, separators[1:]))
                current = p
            else:
                current += p

        if current:
            chunks.extend(self._split_recursive(current, separators[1:]))

        return chunks

    def _chunk_text(self, text: str) -> list[str]:
        """强制切分成定长块，带 overlap"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
        return chunks


BreakpointThresholdType = Literal[
    "percentile", "standard_deviation", "interquartile", "gradient"
]
BREAKPOINT_DEFAULTS: Dict[BreakpointThresholdType, float] = {
    "percentile": 95,
    "standard_deviation": 3,
    "interquartile": 1.5,
    "gradient": 95,
}
Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]

def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices."""
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {X.shape} "
            f"and Y has shape {Y.shape}."
        )
    try:
        import simsimd as simd
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        Z = 1 - np.array(simd.cdist(X, Y, metric="cosine"))
        return Z
    except ImportError:
        X_norm = np.linalg.norm(X, axis=1)
        Y_norm = np.linalg.norm(Y, axis=1)
        # Ignore divide by zero errors run time warnings as those are handled below.
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
        similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
        return similarity


def combine_sentences(sentences: List[dict], buffer_size: int = 1) -> List[dict]:
    for i in range(len(sentences)):
        combined_sentence = ""
        for j in range(i - buffer_size, i):
            if j >= 0:
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += sentences[j]["sentence"] + " "
        combined_sentence += sentences[i]["sentence"]

        for j in range(i + 1, i + 1 + buffer_size):
            # Check if the index j is within the range of the sentences list
            if j < len(sentences):
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += " " + sentences[j]["sentence"]
        sentences[i]["combined_sentence"] = combined_sentence

    return sentences


def calculate_cosine_distances(sentences: List[dict]) -> Tuple[List[float], List[dict]]:
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]["combined_sentence_embedding"]
        embedding_next = sentences[i + 1]["combined_sentence_embedding"]

        # Calculate cosine similarity
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]

        # Convert to cosine distance
        distance = 1 - similarity

        # Append cosine distance to the list
        distances.append(distance)

        # Store distance in the dictionary
        sentences[i]["distance_to_next"] = distance
    return distances, sentences


class SemanticChunker:
    def __init__(
        self,
        embeddings: Embeddings,
        buffer_size: int = 1,
        add_start_index: bool = False,
        breakpoint_threshold_type: BreakpointThresholdType = "percentile",
        breakpoint_threshold_amount: Optional[float] = None,
        number_of_chunks: Optional[int] = None,
        sentence_split_regex: str = r"(?<=[.?!])\s+",
        min_chunk_size: Optional[int] = None,
    ):
        self._add_start_index = add_start_index
        self.embeddings = embeddings
        self.buffer_size = buffer_size
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.number_of_chunks = number_of_chunks
        self.sentence_split_regex = sentence_split_regex
        if breakpoint_threshold_amount is None:
            self.breakpoint_threshold_amount = BREAKPOINT_DEFAULTS[
                breakpoint_threshold_type
            ]
        else:
            self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.min_chunk_size = min_chunk_size

    def _calculate_breakpoint_threshold(
        self, distances: List[float]
    ) -> Tuple[float, List[float]]:
        if self.breakpoint_threshold_type == "percentile":
            return cast(
                float,
                np.percentile(distances, self.breakpoint_threshold_amount),
            ), distances
        elif self.breakpoint_threshold_type == "standard_deviation":
            return cast(
                float,
                np.mean(distances)
                + self.breakpoint_threshold_amount * np.std(distances),
            ), distances
        elif self.breakpoint_threshold_type == "interquartile":
            q1, q3 = np.percentile(distances, [25, 75])
            iqr = q3 - q1

            return np.mean(
                distances
            ) + self.breakpoint_threshold_amount * iqr, distances
        elif self.breakpoint_threshold_type == "gradient":
            # Calculate the threshold based on the distribution of gradient of distance array. # noqa: E501
            distance_gradient = np.gradient(distances, range(0, len(distances)))
            return cast(
                float,
                np.percentile(distance_gradient, self.breakpoint_threshold_amount),
            ), distance_gradient
        else:
            raise ValueError(
                f"Got unexpected `breakpoint_threshold_type`: "
                f"{self.breakpoint_threshold_type}"
            )

    def _threshold_from_clusters(self, distances: List[float]) -> float:
        if self.number_of_chunks is None:
            raise ValueError(
                "This should never be called if `number_of_chunks` is None."
            )
        x1, y1 = len(distances), 0.0
        x2, y2 = 1.0, 100.0

        x = max(min(self.number_of_chunks, x1), x2)

        # Linear interpolation formula
        if x2 == x1:
            y = y2
        else:
            y = y1 + ((y2 - y1) / (x2 - x1)) * (x - x1)

        y = min(max(y, 0), 100)

        return cast(float, np.percentile(distances, y))

    def _calculate_sentence_distances(
        self, single_sentences_list: List[str]
    ) -> Tuple[List[float], List[dict]]:
        """Split text into multiple components."""

        _sentences = [
            {"sentence": x, "index": i} for i, x in enumerate(single_sentences_list)
        ]
        sentences = combine_sentences(_sentences, self.buffer_size)
        embeddings = self.embeddings.embed_documents(
            [x["combined_sentence"] for x in sentences]
        )
        for i, sentence in enumerate(sentences):
            sentence["combined_sentence_embedding"] = embeddings[i]

        return calculate_cosine_distances(sentences)

    def split_text(
        self,
        text: str,
    ) -> List[str]:
        # Splitting the essay (by default on '.', '?', and '!')
        single_sentences_list = re.split(self.sentence_split_regex, text)

        # having len(single_sentences_list) == 1 would cause the following
        # np.percentile to fail.
        if len(single_sentences_list) == 1:
            return single_sentences_list
        # similarly, the following np.gradient would fail
        if (
            self.breakpoint_threshold_type == "gradient"
            and len(single_sentences_list) == 2
        ):
            return single_sentences_list
        distances, sentences = self._calculate_sentence_distances(single_sentences_list)
        if self.number_of_chunks is not None:
            breakpoint_distance_threshold = self._threshold_from_clusters(distances)
            breakpoint_array = distances
        else:
            (
                breakpoint_distance_threshold,
                breakpoint_array,
            ) = self._calculate_breakpoint_threshold(distances)

        indices_above_thresh = [
            i
            for i, x in enumerate(breakpoint_array)
            if x > breakpoint_distance_threshold
        ]

        chunks = []
        start_index = 0

        for index in indices_above_thresh:

            end_index = index

            group = sentences[start_index : end_index + 1]
            combined_text = " ".join([d["sentence"] for d in group])

            if (
                self.min_chunk_size is not None
                and len(combined_text) < self.min_chunk_size
            ):
                continue
            chunks.append(combined_text)

            # Update the start index for the next group
            start_index = index + 1
        if start_index < len(sentences):
            combined_text = " ".join([d["sentence"] for d in sentences[start_index:]])
            chunks.append(combined_text)
        return chunks


   