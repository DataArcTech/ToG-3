"""Data model package for RAG Factory"""

from .document import Document
from .search_result import SearchResult
from .retrieval_result import RetrievalResult

__all__ = ["Document", "SearchResult", "RetrivealResult"]