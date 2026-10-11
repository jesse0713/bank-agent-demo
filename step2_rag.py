"""Step 2：RAG 知识库 —— 文档切片、向量化入库、相似检索"""
import os
import shutil
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import config

COLLECTION_NAME = "bank_products"


def get_embeddings():
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.SILICONFLOW_API_KEY,
        base_url=config.SILICONFLOW_BASE_URL,
    )


def build_vectorstore(force: bool = False):
    """读取 data/products.md，切片后向量化入库；force=True 时重建"""
    if os.path.exists(config.DB_DIR) and os.listdir(config.DB_DIR) and not force:
        print(f"[RAG] 向量库已存在：{config.DB_DIR}（如需重建，传 force=True）")
        return

    if force and os.path.exists(config.DB_DIR):
        shutil.rmtree(config.DB_DIR)

    path = os.path.join(config.DATA_DIR, "products.md")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # 按 "## " 二级标题切分（每个产品一个块），并记录标题为元数据
    blocks = [b.strip() for b in text.split("## ") if b.strip()]
    docs = []
    for b in blocks:
        title = b.split("\n", 1)[0]
        docs.append(Document(page_content=b, metadata={"title": title}))

    # 长文档再按字符细分，便于更精准检索
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.split_documents(docs)

    Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        persist_directory=config.DB_DIR,
        collection_name=COLLECTION_NAME,
    )
    print(f"[RAG] 已入库 {len(docs)} 个片段 → {config.DB_DIR}")


def retrieve(query: str, k: int = 3):
    """检索与 query 最相关的 k 个片段"""
    if not os.path.exists(config.DB_DIR) or not os.listdir(config.DB_DIR):
        raise FileNotFoundError("向量库尚未构建，请先运行 build_vectorstore()")

    vectorstore = Chroma(
        embedding_function=get_embeddings(),
        persist_directory=config.DB_DIR,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore.similarity_search(query, k=k)


if __name__ == "__main__":
    build_vectorstore(force=True)
    for q in ["有没有稳健一点的理财", "大额存单利率多少", "权益类产品风险如何"]:
        print(f"\n查询：{q}")
        for d in retrieve(q, k=2):
            print(f"  - [{d.metadata.get('title', '?')}] {d.page_content[:60]}...")
