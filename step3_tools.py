"""Step 3：工具调用 —— 定义查询产品/利率工具，用 ReAct Agent 调度"""
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
import config
import step2_rag


@tool
def search_product(question: str) -> str:
    """查询银行理财产品/存款产品信息。当用户询问某类产品的收益、风险等级、起购金额、期限时使用。入参是用户的产品问题。"""
    docs = step2_rag.retrieve(question, k=3)
    if not docs:
        return "未找到相关产品信息。"
    return "\n\n".join(f"【{d.metadata.get('title', '产品')}】{d.page_content}" for d in docs)


@tool
def get_deposit_rate(term: str) -> str:
    """查询单个期限的定期存款挂牌利率。term 取值为：'3个月'、'6个月'、'1年'、'2年'、'3年'。"""
    rates = {"3个月": "1.10%", "6个月": "1.30%", "1年": "1.50%", "2年": "1.70%", "3年": "1.95%"}
    r = rates.get(term)
    if r is None:
        return f"暂不支持该期限（{term}），可选：{'、'.join(rates.keys())}"
    return f"{term}定期存款挂牌利率为 {r}（演示数据）。"


@tool
def list_deposit_rates() -> str:
    """一次性列出所有期限的定期存款挂牌利率和大额存单利率。
    当客户想比较利率高低、或询问「有没有利率更高的产品/期限」时使用。"""
    return (
        "【定期存款挂牌利率（整存整取）】\n"
        "3 个月 1.10%　6 个月 1.30%　1 年 1.50%　2 年 1.70%　3 年 1.95%\n\n"
        "【个人大额存单（20 万元起存）】\n"
        "1 年 1.90%　2 年 2.30%　3 年 2.60%\n\n"
        "说明：以上均为存款类产品，受《存款保险条例》保护、保本保息（演示数据）。\n"
        "结论：同等条件下，期限越长利率越高；大额存单利率高于同期普通定期存款。"
    )


def build_agent():
    llm = ChatOpenAI(
        model=config.DEEPSEEK_MODEL,
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        temperature=0.3,
    )
    tools = [search_product, get_deposit_rate, list_deposit_rates]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是银行智能客服助手，回答客户关于存款、理财产品的咨询。必须基于工具返回的真实信息回答，不要编造数据。"
                   "合规口径：存款类产品（定期、活期、大额存单）受《存款保险条例》保护，可以如实说明其保本保息；"
                   "但理财/基金等资管产品不得承诺保本、保收益或无风险。"
                   "当客户问「有没有利率更高的」时，主动用 list_deposit_rates 比较各期限与产品，给出更高利率的具体选项。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, max_iterations=5, verbose=False)


if __name__ == "__main__":
    step2_rag.build_vectorstore()
    agent = build_agent()
    q = input("客户：")
    print("客服：", agent.invoke({"input": q})["output"])
