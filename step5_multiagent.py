"""Step 5：多 Agent 编排 —— 客服 Agent + 合规审查 Agent（LangGraph）"""
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
import config
import step3_tools
import step4_compliance


class State(TypedDict):
    messages: Annotated[list, add_messages]
    draft: str
    verdict: str


COMPLIANCE_PROMPT = (
    "你是银行合规审查员。审查下面这段客服回复是否合规。"
    "违规情形包括：对理财/基金等资管产品承诺收益、承诺保本、声称无风险，"
    "或使用「稳赚不赔、绝对无风险」等绝对化用语，或未提示风险或未做适当性提示。"
    "注意：存款类产品（定期、活期、大额存单）受《存款保险条例》保护，"
    "如实说明其「保本保息」属于合规表述，不算违规。"
    "若合规只回答「合规」，若违规回答「违规：<原因>」。\n\n客服回复：\n{text}"
)


def _build_llm():
    return ChatOpenAI(
        model=config.DEEPSEEK_MODEL,
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        temperature=0.3,
    )


def build_multi_agent():
    llm = _build_llm()
    tools = [step3_tools.search_product, step3_tools.get_deposit_rate, step3_tools.list_deposit_rates]

    # 客服 Agent：ReAct 带工具，负责生成回答草稿
    customer_agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=(
            "你是银行智能客服助手。回答客户关于存款、理财产品的咨询，"
            "必须基于工具返回的真实信息，不要编造数据。"
            "合规口径：存款类产品（定期、活期、大额存单）受《存款保险条例》保护，"
            "可以如实说明其保本保息；但理财/基金等资管产品不得承诺保本、收益或无风险。"
            "当客户问「有没有利率更高的」时，主动用 list_deposit_rates 比较各期限与产品，"
            "给出更高利率的具体选项。"
        ),
    )

    def customer_service(state: State) -> dict:
        result = customer_agent.invoke({"messages": state["messages"]})
        draft = result["messages"][-1].content
        return {"draft": draft}

    def compliance_review(state: State) -> dict:
        draft = state["draft"]
        # 第一层：规则硬拦截（快速、确定性）
        ok, hits = step4_compliance.compliance_check(draft)
        if not ok:
            reply = f"[合规拦截] 命中违规词：{'、'.join(hits)}，已转人工。"
            return {"verdict": reply, "messages": [("ai", reply)]}
        # 第二层：LLM 软审查（判断是否缺失风险/适当性提示）
        review = llm.invoke(COMPLIANCE_PROMPT.format(text=draft)).content
        if "违规" in review:
            reply = f"[合规拦截] {review}"
            return {"verdict": reply, "messages": [("ai", reply)]}
        return {"verdict": "合规", "messages": [("ai", draft)]}

    graph = StateGraph(State)
    graph.add_node("customer_service", customer_service)
    graph.add_node("compliance_review", compliance_review)
    graph.add_edge(START, "customer_service")
    graph.add_edge("customer_service", "compliance_review")
    graph.add_edge("compliance_review", END)
    return graph.compile()


if __name__ == "__main__":
    import step2_rag
    step2_rag.build_vectorstore()
    agent = build_multi_agent()
    q = input("客户：")
    result = agent.invoke({"messages": [("user", q)]})
    # 取最后一条 AI 消息作为最终回复
    for m in reversed(result["messages"]):
        if getattr(m, "type", "") == "ai" and getattr(m, "content", ""):
            print("客服：", m.content)
            break
