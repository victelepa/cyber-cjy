"""
长期记忆 — ChromaDB 向量检索
检索相关历史对话片段和 few-shot 示例
"""
from pathlib import Path


class LongTermMemory:
    def __init__(self, chroma_dir: str, embedding_fn=None, top_k: int = 5):
        self.chroma_dir = chroma_dir
        self.top_k = top_k
        self._client = None
        self._conv_collection = None
        self._pairs_collection = None
        self._embedding_fn = embedding_fn
        self._available = False

    def _init(self):
        if self._client is not None:
            return
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.chroma_dir)
            collections = {c.name for c in self._client.list_collections()}

            kwargs = {}
            if self._embedding_fn:
                kwargs["embedding_function"] = self._embedding_fn

            if "conversations" in collections:
                self._conv_collection = self._client.get_collection(
                    "conversations", **kwargs
                )
            if "few_shot_pairs" in collections:
                self._pairs_collection = self._client.get_collection(
                    "few_shot_pairs", **kwargs
                )
            self._available = bool(self._conv_collection or self._pairs_collection)
        except Exception as e:
            print(f"[长期记忆] 初始化失败：{e}（将跳过向量检索）")
            self._available = False

    @property
    def available(self) -> bool:
        self._init()
        return self._available

    def retrieve_context(self, query: str) -> str:
        """
        检索与当前输入最相关的历史对话片段。
        返回格式化的字符串，直接注入 Prompt。
        """
        self._init()
        if not self._conv_collection:
            return ""

        try:
            results = self._conv_collection.query(
                query_texts=[query],
                n_results=min(self.top_k, self._conv_collection.count()),
                include=["documents", "metadatas"],
            )
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            if not docs:
                return ""

            lines = []
            for doc, meta in zip(docs, metas):
                ts = meta.get("start_time", "")[:10]
                lines.append(f"[{ts}]\n{doc}")
            return "\n\n".join(lines)
        except Exception as e:
            print(f"[长期记忆] 检索失败：{e}")
            return ""

    def retrieve_few_shot(self, query: str, n: int = 3) -> str:
        """
        检索相似场景下她的真实回复，作为 few-shot 示例。
        返回格式化字符串。
        """
        self._init()
        if not self._pairs_collection:
            return ""

        try:
            results = self._pairs_collection.query(
                query_texts=[query],
                n_results=min(n, self._pairs_collection.count()),
                include=["documents", "metadatas"],
            )
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            if not docs:
                return ""

            lines = []
            for doc, meta in zip(docs, metas):
                her_reply = meta.get("output", "")
                if her_reply:
                    lines.append(f"你: {doc}\n她: {her_reply}")
            return "\n\n".join(lines)
        except Exception as e:
            print(f"[长期记忆] few-shot 检索失败：{e}")
            return ""

    def add_exchange(self, user_input: str, assistant_reply: str, timestamp: str = ""):
        """将新的对话轮次存入向量库"""
        self._init()
        if not self._conv_collection:
            return
        try:
            import time
            doc_id = f"live_{int(time.time() * 1000)}"
            text = f"你: {user_input}\n她: {assistant_reply}"
            self._conv_collection.add(
                ids=[doc_id],
                documents=[text],
                metadatas=[{
                    "session_id": "live",
                    "start_time": timestamp,
                    "end_time": timestamp,
                    "chunk_type": "live",
                }],
            )
        except Exception as e:
            print(f"[长期记忆] 存储失败：{e}")
