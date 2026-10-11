"""银行智能客服 Agent Demo —— 整合入口（RAG + 工具调用 + 合规拦截 + 多 Agent）"""
import config
import step2_rag
import step5_multiagent


def main():
    if not config.DEEPSEEK_API_KEY:
        print("❌ 未配置 DEEPSEEK_API_KEY，请先复制 .env.example 为 .env 并填入密钥")
        return
    if not config.SILICONFLOW_API_KEY:
        print("❌ 未配置 SILICONFLOW_API_KEY，请先复制 .env.example 为 .env 并填入密钥")
        return

    print("[init] 构建向量库...")
    step2_rag.build_vectorstore()
    agent = step5_multiagent.build_multi_agent()

    print("=== 银行智能客服 Agent Demo ===（输入 exit 退出）")
    while True:
        q = input("\n客户：").strip()
        if q.lower() in ("exit", "quit", "q"):
            break
        if not q:
            continue
        result = agent.invoke({"messages": [("user", q)]})
        for m in reversed(result["messages"]):
            if getattr(m, "type", "") == "ai" and getattr(m, "content", ""):
                print("客服：", m.content)
                break


if __name__ == "__main__":
    main()
