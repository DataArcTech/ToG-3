import os
from pprint import pprint
from .openai_llm import OpenAILLM  # 你的类所在文件

# ==== 配置 ====
# API_KEY = os.getenv("OPENAI_API_KEY")  # 或直接写成 "sk-xxxx"
API_KEY = "sk-2T06b7c7f9c3870049fbf8fada596b0f8ef908d1e233KLY2"
# BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
BASE_URL = "https://api.gptsapi.net/v1"

def test_openai_llm():
    # 初始化普通对话模型
    llm = OpenAILLM(
        model_name="gpt-4o-mini", 
        api_key=API_KEY, 
        base_url=BASE_URL
    )

    # ==== 1. 测试 chat ====
    messages = [{"role": "user", "content": "你好，请用一句话介绍你自己"}]
    print("\n=== chat (普通) ===")
    result = llm.chat(messages)
    print("回复:", result)

    # ==== 2. 测试 chat + token 统计 ====
    print("\n=== chat (返回 token 数) ===")
    result, token_stats = llm.chat(messages, return_token_count=True)
    print("回复:", result)
    print("token统计:", token_stats)

    # ==== 3. 测试 stream_chat (普通流式) ====
    print("\n=== stream_chat (普通流式) ===")
    for chunk in llm.stream_chat(messages):
        print(chunk, end="", flush=True)
    print()

    # ==== 4. 测试 stream_chat (返回 token 数) ====
    print("\n=== stream_chat (返回 token 数) ===")
    for item in llm.stream_chat(messages, return_token_count=True):
        if isinstance(item, dict) and "input_tokens" in item:
            # 这是最后的token统计信息
            print("\nToken统计:", item)
        else:
            # 这是文本片段
            print(item, end="", flush=True)


    # ==== 7. 测试 get_model_info ====
    print("\n=== get_model_info ===")
    pprint(llm.get_model_info())

    # ==== 8. 测试 get_available_models ====
    print("\n=== get_available_models ===")
    model_list = llm.get_available_models()
    print("可用模型数量:", len(model_list))
    print("前10个模型:", model_list[:10])

    # ==== 9. 测试 embed ====
    # 使用专门的嵌入模型
    embed_llm = OpenAILLM(
        model_name="text-embedding-3-small", 
        api_key=API_KEY, 
        base_url=BASE_URL
    )
    print("\n=== embed (单条) ===")
    vec = embed_llm.embed("这是一个测试文本")
    print("向量维度:", len(vec))

    print("\n=== embed (多条) ===")
    vecs = embed_llm.embed(["第一句", "第二句"])
    print("向量数量:", len(vecs), "每个向量维度:", len(vecs[0]))


if __name__ == "__main__":
    test_openai_llm()
