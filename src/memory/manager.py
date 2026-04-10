"""记忆管理器 — 协调三级记忆系统"""
import yaml
from pathlib import Path

from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory
from src.memory.core_memory import CoreMemory


def _build_embedding_fn(emb_cfg: dict):
    """构建与 build_index.py 一致的 embedding function"""
    provider = emb_cfg.get("provider", "local")
    if provider == "local":
        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            model = emb_cfg.get("model_name", "BAAI/bge-small-zh-v1.5")
            return SentenceTransformerEmbeddingFunction(model_name=model)
        except Exception as e:
            print(f"[记忆] 本地 embedding 加载失败（{e}），使用默认 embedding")
            return None
    elif provider == "openai":
        import os
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
            return OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=emb_cfg.get("openai_model", "text-embedding-3-small"),
            )
    return None


class MemoryManager:
    def __init__(self, config: dict):
        mem_cfg = config.get("memory", {})
        emb_cfg = config.get("embedding", {})

        self.short_term = ShortTermMemory(
            window=mem_cfg.get("short_term_window", 20)
        )
        embedding_fn = _build_embedding_fn(emb_cfg)
        self.long_term = LongTermMemory(
            chroma_dir=mem_cfg.get("chroma_dir", "data/chroma_db"),
            top_k=mem_cfg.get("long_term_top_k", 5),
            embedding_fn=embedding_fn,
        )
        self.core = CoreMemory(
            path=mem_cfg.get("core_memory_path", "data/core_memory.json")
        )
        self._turn_count = 0
        self._core_check_interval = mem_cfg.get("core_check_interval", 3)

    def retrieve(self, user_input: str) -> dict:
        """
        检索所有层级的记忆，返回字典供 Prompt 组装使用。
        """
        return {
            "short_term_messages": self.short_term.get_messages(),
            "retrieved_context": self.long_term.retrieve_context(user_input),
            "few_shot_examples": self.long_term.retrieve_few_shot(user_input, n=2),
            "core_memory": self.core.get_all(),
        }

    def update(self, user_input: str, assistant_reply: str, llm=None):
        """
        对话结束后更新所有层级的记忆。
        """
        from datetime import datetime
        ts = datetime.now().isoformat()

        # 1. 更新短期记忆
        self.short_term.add_exchange(user_input, assistant_reply)

        # 2. 更新长期记忆（向量库）
        self.long_term.add_exchange(user_input, assistant_reply, ts)

        # 3. 每 N 轮检查一次核心记忆（节省 API 开销）
        self._turn_count += 1
        if llm and self._turn_count % self._core_check_interval == 0:
            self.core.try_extract_and_update(llm, user_input, assistant_reply)

    def reset_short_term(self):
        self.short_term.reset()
