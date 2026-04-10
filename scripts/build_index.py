"""
Phase 4 构建 ChromaDB 向量索引

将预处理后的对话对和对话会话 embedding 存入 ChromaDB，
供长期记忆检索使用。

使用方式：
  python scripts/build_index.py

默认使用 ChromaDB 内置 embedding（sentence-transformers，无需 API key）。
中文效果一般但够用，后续可换成 BGE 或 OpenAI embedding。

索引内容：
  - few_shot_pairs.jsonl 中的对话对（"你说→她回"，用于 few-shot 检索）
  - conversations/ 中的对话会话（用于上下文检索）
"""
import json
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

PAIRS_PATH = "data/processed/few_shot_pairs.jsonl"
CONV_DIR = "data/processed/conversations"
CHROMA_DIR = "data/chroma_db"
CONFIG_PATH = "config.yaml"

CONV_CHUNK_SIZE = 8      # 每个会话 chunk 包含的消息数
CONV_OVERLAP = 2         # chunk 之间的重叠消息数


def get_embedding_fn(config: dict):
    """
    返回 ChromaDB 兼容的 embedding function。
    优先使用本地 sentence-transformers（免费），
    config 中可切换为 OpenAI embedding。
    """
    provider = config.get("provider", "local")

    if provider == "openai":
        import os
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            print("  未找到 OPENAI_API_KEY，回退到本地 embedding")
            provider = "local"
        else:
            return OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=config.get("openai_model", "text-embedding-3-small"),
            )

    if provider == "local":
        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            model = config.get("model_name", "BAAI/bge-small-zh-v1.5")
            print(f"  使用本地 embedding 模型：{model}（首次运行会自动下载）")
            return SentenceTransformerEmbeddingFunction(model_name=model)
        except Exception as e:
            print(f"  本地 embedding 加载失败（{e}），使用 ChromaDB 默认 embedding")
            return None  # ChromaDB 默认 all-MiniLM-L6-v2

    return None


def load_pairs() -> list[dict]:
    pairs = []
    with open(PAIRS_PATH, encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def load_conversations() -> list[tuple[str, list[dict]]]:
    """加载所有对话会话，返回 [(session_id, messages), ...]"""
    sessions = []
    conv_dir = Path(CONV_DIR)
    files = sorted(conv_dir.glob("conv_*.jsonl"))
    for f in files:
        msgs = []
        with open(f, encoding="utf-8") as fp:
            for line in fp:
                msgs.append(json.loads(line))
        if msgs:
            session_id = f.stem
            sessions.append((session_id, msgs))
    return sessions


def chunk_conversation(session_id: str, messages: list[dict],
                       chunk_size: int = 8, overlap: int = 2) -> list[dict]:
    """将一个对话会话切分为若干 chunk，每个 chunk 是一段连续消息"""
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(messages), step):
        chunk_msgs = messages[i:i + chunk_size]
        if not chunk_msgs:
            continue
        text = "\n".join(
            f"{m['sender']}: {m['content']}"
            for m in chunk_msgs
            if m.get("content") and m.get("type") in ("text", "voice")
        )
        if len(text.strip()) < 10:
            continue
        chunks.append({
            "id": f"{session_id}_chunk_{i // step:04d}",
            "text": text,
            "metadata": {
                "session_id": session_id,
                "start_time": chunk_msgs[0].get("timestamp", ""),
                "end_time": chunk_msgs[-1].get("timestamp", ""),
                "chunk_type": "conversation",
            }
        })
    return chunks


def build_pairs_collection(client, embedding_fn, pairs: list[dict]):
    """构建对话对集合（few-shot 检索用）"""
    coll_kwargs = {"name": "few_shot_pairs", "metadata": {"hnsw:space": "cosine"}}
    if embedding_fn:
        coll_kwargs["embedding_function"] = embedding_fn

    try:
        client.delete_collection("few_shot_pairs")
    except Exception:
        pass
    collection = client.create_collection(**coll_kwargs)

    batch_size = 500
    total = len(pairs)
    for i in range(0, total, batch_size):
        batch = pairs[i:i + batch_size]
        ids = [f"pair_{i + j}" for j in range(len(batch))]
        # 用"你说的话"作为检索 key（查询时也用用户输入检索）
        documents = [p["input"] for p in batch]
        metadatas = [
            {
                "output": p["output"][:500],   # 她的回复（截断避免太长）
                "timestamp": p.get("timestamp", ""),
            }
            for p in batch
        ]
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        print(f"  对话对：{min(i + batch_size, total)}/{total}")

    return collection


def build_conversations_collection(client, embedding_fn, sessions: list[tuple]):
    """构建对话会话集合（上下文检索用）"""
    coll_kwargs = {"name": "conversations", "metadata": {"hnsw:space": "cosine"}}
    if embedding_fn:
        coll_kwargs["embedding_function"] = embedding_fn

    try:
        client.delete_collection("conversations")
    except Exception:
        pass
    collection = client.create_collection(**coll_kwargs)

    all_chunks = []
    for session_id, messages in sessions:
        chunks = chunk_conversation(session_id, messages, CONV_CHUNK_SIZE, CONV_OVERLAP)
        all_chunks.extend(chunks)

    batch_size = 500
    total = len(all_chunks)
    for i in range(0, total, batch_size):
        batch = all_chunks[i:i + batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )
        print(f"  对话 chunk：{min(i + batch_size, total)}/{total}")

    return collection, total


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    print("初始化 embedding 模型...")
    embedding_fn = get_embedding_fn(config.get("embedding", {}))

    # 构建对话对索引
    print("\n构建对话对索引（few-shot 检索）...")
    pairs = load_pairs()
    print(f"  加载 {len(pairs)} 个对话对")
    build_pairs_collection(client, embedding_fn, pairs)
    print(f"  完成")

    # 构建对话会话索引
    print("\n构建对话会话索引（上下文检索）...")
    sessions = load_conversations()
    print(f"  加载 {len(sessions)} 个会话")
    _, total_chunks = build_conversations_collection(client, embedding_fn, sessions)
    print(f"  完成，共 {total_chunks} 个 chunk")

    print(f"\n索引已保存到 {CHROMA_DIR}")
    print("下一步：重启 app.py，长期记忆系统将自动启用")


if __name__ == "__main__":
    main()
