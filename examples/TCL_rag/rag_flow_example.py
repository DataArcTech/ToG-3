import sys
import os

# 添加 RAG-Factory 目录到 Python 路径
rag_factory_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, rag_factory_path)

from rag_factory.llms import LLMRegistry
from rag_factory.Embed import EmbeddingRegistry
from rag_factory.Store import VectorStoreRegistry
from rag_factory.Retrieval import RetrieverRegistry
from rag_factory.rerankers import RerankerRegistry
from rag_factory.Retrieval import Document
from prompt import ANALYZE_GRAPH_PROMPT, ANALYZE_RAG_PROMPT, ANALYZE_LLM_PROMPT
from typing import List
import json


class TCL_RAG:
    def __init__(
        self,
        *,
        llm_config=None,
        embedding_config=None,
        vector_store_config=None,
        bm25_retriever_config=None,
        retriever_config=None,
        reranker_config=None,
    ):
        llm_config = llm_config or {}
        embedding_config = embedding_config or {}
        vector_store_config = vector_store_config or {}
        bm25_retriever_config = bm25_retriever_config or {}
        retriever_config = retriever_config or {}
        reranker_config = reranker_config or {}
        self.llm = LLMRegistry.create(**llm_config)
        self.embedding = EmbeddingRegistry.create(**embedding_config)
        self.vector_store = VectorStoreRegistry.load(**vector_store_config, embedding=self.embedding)
        self.bm25_retriever = RetrieverRegistry.create(**bm25_retriever_config)
        self.bm25_retriever = self.bm25_retriever.from_documents(documents=self._load_data(bm25_retriever_config["data_path"]), preprocess_func=self.chinese_preprocessing_func, k=bm25_retriever_config["k"])

        self.retriever = RetrieverRegistry.create(**retriever_config, vectorstore=self.vector_store)
        self.multi_path_retriever = RetrieverRegistry.create("multipath", retrievers=[self.bm25_retriever, self.retriever])
        self.reranker = RerankerRegistry.create(**reranker_config)

    def invoke(self, query: str, k: int = None):
        return self.multi_path_retriever.invoke(query, top_k=k)

    def rerank(self, query: str, documents: List[Document], k: int = None, batch_size: int = 8):
        return self.reranker.rerank(query, documents, k, batch_size)

    def _load_data(self, data_path: str):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            docs = []
            for item in data:
                content = item.get("content", "")
                metadata = item.get("metadata", {})
                docs.append(Document(content=content, metadata=metadata))
        return docs

    def chinese_preprocessing_func(self, text: str) -> str:
        import jieba
        return " ".join(jieba.cut(text))
    
    # def rewrite_query(self, query: str):
    #     template = (
    #         "你是一位行测领域的专家，擅长根据用户问题中的背景资料和具体题目及选项，重写为适合信息检索的query。\n"
    #         "遵循以下要求：\n"
    #         "1. 提取背景材料中的核心事实和关键数据。\n"
    #         "2. 明确题目和选项的考查知识点和所需信息。\n"
    #         "3. 依据考查知识点和所需信息，从背景资料中提取相关信息。\n"
    #         "4. 背景资料中缺少的信息，需要通过背景资料中的其他信息推算。\n"
    #         "5. 使用简洁的陈述句式而非疑问句。\n"
    #         "6. 保留专业术语和关键数字。\n"
    #         "7. 重写后的query需要简洁明了，便于信息检索，只包含解答问题所需的参考信息。\n"
    #         "8. 答复只包含重写后的query，不要包含其他内容。\n"
    #         "用户问题：{question}\n"
    #         "答复："
    #     )
    #     prompt = template.format(question=query)
    #     messages = [
    #         {"role": "system", "content": "你是一位行测领域的专家，擅长根据用户问题中的背景资料和具体题目及选项，重写为适合信息检索的query。"},
    #         {"role": "user", "content": prompt}
    #     ]
    #     return self.llm.chat(messages, temperature=0.2)

    def rewrite_query(self, query: str):
        template = (
            "你是一位行测领域的专家，擅长根据用户问题中的背景资料和具体题目及选项，提取这道题目的考查知识点。\n"
            "遵循以下要求：\n"
            "1. 仔细阅读题目内容，包括材料和数据。\n"
            "2. 识别题目考查的具体计算能力和分析技能。\n"
            "3. 提取核心知识点并分类。\n"
            "4. 答复只包含核心知识点，不要包含其他内容。\n"
            "用户问题：{question}\n"
            "答复："
        )
        prompt = template.format(question=query)
        messages = [
            {"role": "system", "content": "你是一位行测领域的专家，擅长根据用户问题中的背景资料和具体题目及选项，提取这道题目的考查知识点。"},
            {"role": "user", "content": prompt}
        ]
        return self.llm.chat(messages, temperature=0.2)
    
    def mark_knowledge(self, query: str):
        template = (
            "你是一位行测领域的专家，擅长根据用户问题中的背景资料和具体题目及选项，标记出解答这道题目所需要的知识点。\n"
            "遵循以下要求：\n"
            "1. 根据题目和选项，总结出解答这道题目所需要的信息。\n"
            "2. 根据解答题目所需要的信息，从背景资料中提取相关信息，如果背景资料中没有，则判断需要通过哪些信息推算。\n"
            "3. 汇总题目和推算所需要的知识点，答复中仅输出这些知识点。\n"
            "4. 知识点需要简洁明了，便于理解。\n"
            "用户问题：{question}\n"
            "答复："
        )
        prompt = template.format(question=query)
        messages = [
            {"role": "system", "content": "你是一位行测领域的专家，擅长根据用户问题中的背景资料和具体题目及选项，标记出用户问题中需要参考的背景资料中的知识点。"},
            {"role": "user", "content": prompt}
        ]
        return self.llm.chat(messages, temperature=0.2)

    def graph_answer(self, query: str, documents: List[Document]):

        # template = (
        #     "你是一位行测领域的专家，擅长根据背景资料得出选择题的答案，并给出解题思路。\n"
        #     "用户问题中包含背景资料和具体题目，主要根据背景资料中的信息，参考以下检索到的参考材料中的例题解题思路、概念、公式等，不可使用参考材料中的数据，回答用户问题中的题目，并给出解析。\n"
        #     "注重通过对背景资料中信息的分析和推理解答题目，禁止根据外部资料和参考材料中的数据解题，参考材料主要用于辅助解题和提供思路参考，需要充分利用背景资料中的信息并保证计算严谨，结论需要有计算结果支持。\n"
        #     "题目中如果提到“根据上文”，指的是背景资料，如果提到“根据上图”或“根据上表”，指的是背景资料中的最后一个图表。\n"
        #     "题目中计算所需要的数据和信息如果无法从背景资料获得，则需要考虑通过背景资料中的其他信息推算，循环该过程，直到可以获得所有所需数据和信息。\n"
        #     "如果选项的逐项分析结果超过一个可以满足用户问题，对所有满足的选项重新分析，给出最终答案。\n"
        #     "如果有用到参考材料中的解题思路、概念、公式等，请在答案中注明使用到的内容摘要及来源title。\n"
        #     "答复严格以json格式输出，注意json结构中使用英文标点符号。\n"
        #     "答复格式：\n"
        #     "1. 解题思路：包含背景资料中的重点信息，参考材料，选项的逐项分析，汇总分析，按照字符串格式输出，不要有json无法解析的符号。json字段为solution\n"
        #     "2. 最终答案：根据解题思路和用户问题，给出最终选项字母编号，如果无法明确确定唯一答案，返回“无法确定”。json字段为answer\n"
        #     "3. 知识点：根据问题和解题思路，结合参考材料，得出该题目考核的核心知识点列表。json字段为knowledge，内容为list格式\n"
        #     "用户问题：{question}\n"
        #     "参考材料：{context}\n"
        #     "答复："
        # )
        context = "\n".join([doc.content for doc in documents])
        # prompt = template.format(question=query, context=context)
        prompt = ANALYZE_GRAPH_PROMPT.format(question=query, context=context)
        messages = [
            {"role": "system", "content": "你是一位行测领域的专家，擅长使用思维链方式逐步推理，根据背景资料得出选择题的答案。"},
            {"role": "user", "content": prompt}
        ]
        return self.llm.chat(messages, temperature=0.0)
    
    def rag_answer(self, query: str, documents: List[Document]):

        # template = (
        #     "你是一位行测领域的专家，擅长根据背景资料得出选择题的答案，并给出解题思路。\n"
        #     "用户问题中包含背景资料和具体题目，主要根据背景资料中的信息，参考以下检索到的参考材料中的例题解题思路、概念、公式等，不可使用参考材料中的数据，回答用户问题中的题目，并给出解析。\n"
        #     "注重通过对背景资料中信息的分析和推理解答题目，禁止根据外部资料和参考材料中的数据解题，参考材料主要用于辅助解题和提供思路参考，需要充分利用背景资料中的信息并保证计算严谨，结论需要有计算结果支持。\n"
        #     "题目中如果提到“根据上文”，指的是背景资料，如果提到“根据上图”或“根据上表”，指的是背景资料中的最后一个图表。\n"
        #     "题目中计算所需要的数据和信息如果无法从背景资料获得，则需要考虑通过背景资料中的其他信息推算，循环该过程，直到可以获得所有所需数据和信息。\n"
        #     "如果选项的逐项分析结果超过一个可以满足用户问题，对所有满足的选项重新分析，给出最终答案。\n"
        #     "如果有用到参考材料中的解题思路、概念、公式等，请在答案中注明使用到的内容摘要及来源title。\n"
        #     "答复严格以json格式输出，注意json结构中使用英文标点符号。\n"
        #     "答复格式：\n"
        #     "1. 解题思路：包含背景资料中的重点信息，参考材料，选项的逐项分析，汇总分析，按照字符串格式输出。json字段为solution\n"
        #     "2. 最终答案：根据解题思路和用户问题，给出最终选项字母编号，如果无法明确确定唯一答案，返回“无法确定”。json字段为answer\n"
        #     "3. 知识点：根据问题和解题思路，结合参考材料，得出该题目考核的核心知识点列表。json字段为knowledge，内容为list格式\n"
        #     "用户问题：{question}\n"
        #     "参考材料：{context}\n"
        #     "答复："
        # )
        context = "\n".join([doc.content for doc in documents])
        # prompt = template.format(question=query, context=context)
        prompt = ANALYZE_RAG_PROMPT.format(question=query, context=context)
        messages = [
            {"role": "system", "content": "你是一位行测领域的专家，擅长使用思维链方式逐步推理，根据背景资料得出选择题的答案。"},
            {"role": "user", "content": prompt}
        ]
        return self.llm.chat(messages, temperature=0.0)

    def llm_answer(self, query: str):
        # template = (
        #     "你是一位行测领域的专家，擅长根据背景资料得出选择题的答案，并给出解题思路。\n"
        #     "用户问题中包含背景资料和具体题目，主要根据背景资料中的信息，回答用户问题中的题目，并给出解析。\n"
        #     "注重通过对背景资料中信息的分析和推理解答题目，禁止根据外部资料解题，需要充分利用背景资料中的信息并保证计算严谨，结论需要有计算结果支持。\n"
        #     "题目中如果提到“根据上文”，指的是背景资料，如果提到“根据上图”或“根据上表”，指的是背景资料中的最后一个图表。\n"
        #     "题目中计算所需要的数据和信息如果无法从背景资料获得，则需要考虑通过背景资料中的其他信息推算，循环该过程，直到可以获得所有所需数据和信息。\n"
        #     "如果选项的逐项分析结果超过一个可以满足用户问题，对所有满足的选项重新分析，给出最终答案。\n"
        #     "答复严格以json格式输出，注意json结构中使用英文标点符号。\n"
        #     "答复格式：\n"
        #     "1. 解题思路：包含背景资料中的重点信息，选项的逐项分析，汇总分析，按照字符串格式输出。json字段为solution\n"
        #     "2. 最终答案：根据解题思路和用户问题，给出最终选项字母编号，如果无法明确确定唯一答案，返回“无法确定”。json字段为answer\n"
        #     "3. 知识点：根据问题和解题思路，得出该题目考核的核心知识点列表。json字段为knowledge，内容为list格式\n"
        #     "用户问题：{question}\n"
        #     "答复："
        # )
        # prompt = template.format(question=query)
        prompt = ANALYZE_LLM_PROMPT.format(question=query)
        messages = [
            {"role": "system", "content": "你是一位行测领域的专家，擅长使用思维链方式逐步推理，根据背景资料得出选择题的答案。"},
            {"role": "user", "content": prompt}
        ]
        return self.llm.chat(messages, temperature=0.0)

    def judge_answer(self, query: str, answer1: str, answer2: str, answer3: str, answer: str, explanation: str):
        template = (
            "你是一位行测领域的教育专家，擅长针对用户问题中的选择题和背景资料，基于这道题目的正确答案和答案解析，评估两个答案哪个更正确。\n"
            "用户问题中包含背景资料和具体题目。\n"
            "打分方式：\n"
            "1. 解题思路中四个选项每个选项的打分范围为0-2分，0分表示完全错误，1分表示思路正确但结果错误，2分表示完全正确。\n"
            "2. 对整体分析过程的打分范围为0-2分，评估答案的逻辑性（解题思路是否逻辑清晰）和可读性（解题思路是否便于理解）。\n"
            "3. 最终打分将解题思路和整体分析过程的打分相加得到。范围为0-10分，如果不在范围内，找到错误原因并重新打分。\n"
            "4. 如果两个答案打分相同，请评估哪个答案的解题思路更接近正确答案。\n"
            "输出要求：\n"
            "1. 请以json格式输出，便于展示。\n"
            "2. 包含答案是否正确，json字段为answer1_is_correct、answer2_is_correct和answer3_is_correct\n"
            "3. 包含每个答案的打分（1-10 分，分数越高表示答案质量越好），每个答案的打分字段为answer1_score、answer2_score和answer3_score\n"
            "4. 包含对每个答案的评估内容（说明打分理由），每个答案的评估内容字段为answer1_reason、answer2_reason和answer3_reason\n"
            "5. 包含最终推荐的答案数字编号（仅推荐一个最优答案，优先推荐正确答案，至少推荐其中一个答案），json字段为recommend\n"
            "6. 包含答案的核心差异和推荐理由，json字段为difference\n"
            "用户问题：{question}\n"
            "正确答案：{answer}\n"
            "答案解析：{explanation}\n"
            "答案1：{answer1}\n"
            "答案2：{answer2}\n"
            "答案3：{answer3}\n"
            "评估结果："
        )
        prompt = template.format(question=query, answer=answer, explanation=explanation, answer1=answer1, answer2=answer2, answer3=answer3)
        messages = [
            {"role": "system", "content": "你是一位行测领域的教育专家，擅长针对用户问题中的选择题和背景资料，基于这道题目的正确答案和解析，评估答案和解题思路。"},
            {"role": "user", "content": prompt}
        ]
        return self.llm.chat(messages, temperature=0.2)
    
    def match_knowledge(self, graph_knowledge: list, rag_knowledge: list, llm_knowledge: list, knowledge: list):
        template = (
            "你是一个专业的字符串分析与模式匹配专家。你的核心任务是精确判断查询字符串列表中的元素是否包含在目标字符串列表中，并计算命中率。你必须具备强大的消歧能力，能处理同义词、缩写、拼写变体和上下文歧义。\n"
            "查询字符串是knowledge，目标字符串是graph_knowledge、rag_knowledge、llm_knowledge。\n"
            "如果查询字符串列表中的元素包含在目标字符串列表中，则算命中，如果查询字符串列表中的元素包含在目标字符串列表中的元素同义词中，则算命中，如果目标字符串列表中的元素包含在查询字符串列表中的元素同义词中，则算命中。\n"
            "查询字符串中只要有一个元素命中，则命中率为100，否则命中率为0。\n"
            "输出要求：\n"
            "1. 请以json格式输出，便于展示。\n"
            "2. 包含目标字符串graph_knowledge、rag_knowledge和llm_knowledge在查询字符串knowledge中的命中率，范围为0-100，json字段为graph_hit_rate、rag_hit_rate和llm_hit_rate\n"
            "graph_knowledge：{graph_knowledge}\n"
            "rag_knowledge：{rag_knowledge}\n"
            "llm_knowledge：{llm_knowledge}\n"
            "knowledge：{knowledge}\n"
            "答复："
        )
        prompt = template.format(graph_knowledge=graph_knowledge, rag_knowledge=rag_knowledge, llm_knowledge=llm_knowledge, knowledge=knowledge)
        messages = [
            {"role": "system", "content": "你是一个专业的字符串分析与模式匹配专家。你的核心任务是精确判断查询字符串列表中的元素是否包含在目标字符串列表中，并计算命中率。你必须具备强大的消歧能力，能处理同义词、缩写、拼写变体和上下文歧义。"},
            {"role": "user", "content": prompt}
        ]
        return self.llm.chat(messages, temperature=0.2)

def _main():
    query = "例题精讲 例1 (2024国考市地) 2022年, 京津冀地区生产总值合计10.0万亿元, 是2013年的1.8倍。其中, 北京、河北跨越4万亿元量级, 均为4.2万亿元, 分别是2013年的2.0倍和1.7倍; 天津1.6万亿元, 是2013年的1.6倍, 京津冀第一产业、第二产业、第三产业增加值占生产总值比重构成由2013年的6.2:35.7:58.1变化为2022年的4.8:29.6:65.6。京津冀三地第三产业增加值占生产总值比重分别为83.8%、61.3%和49.4%, 较2013年分别提高4.3、7.2和8.4个百分点。2013年, 北京第三产业增加值占其生产总值比重比天津高多少个百分点?\nA. 25.4\nB. 22.5\nC. 19.6\nD. 16.7"
    template = (
            "你是一位行测领域的专家，擅长根据背景资料得出选择题的答案，并给出解题思路。\n"
            "用户问题中包含背景资料和具体题目，主要根据背景资料中的信息，回答用户问题中的题目，并给出解析。\n"
            "注重通过对背景资料中信息的分析和推理解答题目，禁止根据外部资料解题，需要充分利用背景资料中的信息并保证计算严谨，结论需要有计算结果支持。\n"
            "题目中如果提到“根据上文”，指的是背景资料，如果提到“根据上图”或“根据上表”，指的是背景资料中的最后一个图表。\n"
            "题目中计算所需要的数据和信息如果无法从背景资料获得，则需要考虑通过背景资料中的其他信息推算，循环该过程，直到可以获得所有所需数据和信息。\n"
            "如果选项的逐项分析结果超过一个可以满足用户问题，对所有满足的选项重新分析，给出最终答案。\n"
            "答复严格以json格式输出，注意json结构中使用英文标点符号。\n"
            "答复格式：\n"
            "1. 解题思路：包含背景资料中的重点信息，选项的逐项分析，汇总分析，按照字符串格式输出。json字段为solution\n"
            "2. 最终答案：根据解题思路和用户问题，给出最终选项字母编号，如果无法明确确定唯一答案，返回“无法确定”。json字段为answer\n"
            "3. 知识点：根据问题和解题思路，得出该题目考核的核心知识点列表。json字段为knowledge，内容为list格式\n"
            "用户问题：{question}\n"
            "答复："
        )
    prompt = template.format(question=query)
    messages = [
        {"role": "system", "content": "你是一位行测领域的专家，擅长根据背景资料得出选择题的答案，并给出解题思路。"},
        {"role": "user", "content": prompt}
    ]
    print(llm.chat(messages, temperature=0.0))


if __name__ == "__main__":
    _main()

