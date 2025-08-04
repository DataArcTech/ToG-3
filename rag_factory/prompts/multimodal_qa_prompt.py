
from llama_index.core.prompts.base import PromptTemplate

qa_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)
MULTIMODAL_QA_TMPL = PromptTemplate(qa_tmpl_str)