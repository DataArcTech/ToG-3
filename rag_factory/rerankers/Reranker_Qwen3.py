import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .Reranker_Base import RerankerBase
from ..Retrieval.RetrieverBase import Document

class Qwen3Reranker(RerankerBase):
    def __init__(self, model_name_or_path: str, max_length: int = 4096, instruction=None, attn_type='causal', device_id="cuda:0", **kwargs):
        super().__init__()
        device = torch.device(device_id)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, padding_side='left')
        self.lm = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.float16)
        self.lm = self.lm.to(device).eval()
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        self.instruction = instruction or "Given the user query, retrieval the relevant passages"
        self.device = device

    def format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = self.instruction
        output = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
        return output

    def process_inputs(self, pairs):
        out = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(out['input_ids']):
            out['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        out = self.tokenizer.pad(out, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in out:
            out[key] = out[key].to(self.lm.device)
        return out

    @torch.no_grad()
    def compute_logits(self, inputs, **kwargs):
        batch_scores = self.lm(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def compute_scores(self, pairs, instruction=None, **kwargs):
        pairs = [self.format_instruction(instruction, query, doc) for query, doc in pairs]
        inputs = self.process_inputs(pairs)
        scores = self.compute_logits(inputs)
        return scores

    def rerank(self, query: str, documents: list[Document], k: int = None, batch_size: int = 8, **kwargs) -> list[Document]:
        # 1. 组装 (query, doc.content) 对
        pairs = [(query, doc.content) for doc in documents]

        # 2. 计算分数
        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            batch_scores = self.compute_scores(batch_pairs)
            all_scores.extend(batch_scores)
        scores = all_scores

        # 3. 按分数排序
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, score in doc_score_pairs]
        if k is not None:
            reranked_docs = reranked_docs[:k]
        return reranked_docs