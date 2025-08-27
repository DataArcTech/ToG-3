import re
import numpy as np

from rag_factory.Embed.Embedding_Base import Embeddings
from tqdm import tqdm
from typing import Literal, Optional, Union, List, Dict, Tuple, cast

  

class LayoutSpliter:
    """
        适用于dots ocr 解析的layout文件
    """
    def __init__(self, 
                begin_pattern = r'^([(（]材料|\d+[)、.]|（材料|[一二三四五六七八九十]+、|（[一二三四五六七八九十]+）|\d+、|\d+. |阅读下列材料|统计表:)'
                ):
        self.begin_pattern = begin_pattern

        
    @staticmethod
    def _lv1_heading(paragraph):
        return bool(re.match(r'^第[一二三四五六七八九十零百千万]+[章]', paragraph.strip()))  
    
    @staticmethod
    def _lv2_heading(paragraph):
        return bool(re.match(r'^第[一二三四五六七八九十零百千万]+[节]', paragraph.strip()))
    
    def split_text_with_title(self, data) -> List:
        """
            对有固定一级标题，二级标题格式的数据进行分块
            输出metadata格式：
            {
                "title_lv1":,
                "title_lv2":,
                "table":,
                "figure":,
                "text":
            }
        """
        title_lv1 = ''
        title_lv2 = ''
        temp_text = ''
        table = []
        figure = []
        meta_data = []
        for index, row in enumerate(tqdm(data)):
            category = row.get('category','')
            text = row.get('text','')
            if index == len(data):
                meta_data.append({
                        "title_lv1":title_lv1,
                        "title_lv2":title_lv2,
                        "table":table,
                        "figure":figure,
                        "text":temp_text
                    })
            if category in ['Page-footer', 'Page-header'] :
                continue
            if category in ["Section-header", "Title"]:
                if len(temp_text) >= 10:
                    meta_data.append({
                        "title_lv1":title_lv1,
                        "title_lv2":title_lv2,
                        "table":table,
                        "figure":figure,
                        "text":temp_text
                    })
                    temp_text = ''
                    table = []
                    figure = []

                text = re.sub(r'^[#*]+\s*', '', text, flags=re.MULTILINE)
                if self._lv1_heading(text):
                    title_lv1 = text
                    title_lv2 = ''
                elif self._lv2_heading(text):
                    title_lv2 = text
                else:
                    temp_text +=" "+text
            elif category == 'Table':
                table.append(text)
                temp_text += text
            elif category == 'Picture':
                if text != '':
                    figure.append(f"page_{row['page_no']}_{row['index']}")
                    temp_text += f"<figure> {text} </figure>"
            else:
                text = re.sub(r'^[#*]+\s*', '', text)
                if re.match(r'^\s*(?:.?)?(例|【例|（例|【解析|典型真题)\s*', text.strip()) and len(temp_text)>5: 
                    meta_data.append({
                        "title_lv1":title_lv1,
                        "title_lv2":title_lv2,
                        "table":table,
                        "figure":figure,
                        "text":temp_text
                    })
                    temp_text = text
                    table = []
                    figure = []
                else:
                    temp_text += " "+text
        return meta_data
    
    def split_text(self, data) -> List[str]:
        """
            将每个标题间的内容分成一个块
        """
        temp_text = ''
        temp = []
        for row in data:
            category = row.get('category','')
            text = str(row.get("text","")).lstrip('#').strip()
            if category == 'Page-header' or category == 'Page-footer':
                continue
            
            if category == 'Picture':
                p_dir = f'page_{row.get("page_no","")}_{row.get("index","")}'
                text = f"<figure> ({p_dir}) {text} </figure>"
            if category in ["Section-header", 'Title']:
                temp.append(temp_text)
                temp_text = text
                continue
            for t in text.split("\n\n"):
                if re.match(self.begin_pattern, t.lstrip("*#").strip()) :
                    temp.append(temp_text)
                    temp_text = t
                else:
                    temp_text += " "+ t
        
        temp.append(temp_text)
        temp = [x for x in temp if x != ""]
        return temp


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
        self._separators = separators or ["\n\n", "\n", " "]
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


   