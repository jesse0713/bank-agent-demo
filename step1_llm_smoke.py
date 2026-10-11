"""Step 1：冒烟测试 —— 验证能调通大模型"""
from langchain_openai import ChatOpenAI
import config


def main():
    llm = ChatOpenAI(
        model=config.DEEPSEEK_MODEL,
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        temperature=0.3,
    )
    resp = llm.invoke("用一句话解释什么是智能投顾")
    print(resp.content)


if __name__ == "__main__":
    main()