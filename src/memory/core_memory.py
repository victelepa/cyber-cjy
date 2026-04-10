"""
核心记忆 — 关键事实 KV 存储
始终注入 System Prompt，由 LLM 在对话中自动更新。
"""
import json
from pathlib import Path
from datetime import datetime


# LLM 用来判断是否需要更新核心记忆的 Prompt
EXTRACT_FACTS_PROMPT = """\
请分析以下对话，判断其中是否出现了值得长期记住的新事实。

"值得记住"的定义：
- 她提到的重要个人信息（生日、家乡、专业、工作等）
- 你们之间的重要约定或承诺
- 她的明确偏好或禁忌（喜欢/讨厌某食物、某活动等）
- 重要的情感节点（吵架和好、表白、纪念日等）

对话：
你: {user_input}
她: {assistant_reply}

如果有新事实，输出 JSON 格式：{{"key": "事实描述"}}
如果没有值得更新的新事实，输出：null

只输出 JSON 或 null，不要有其他文字。"""


class CoreMemory:
    def __init__(self, path: str):
        self.path = Path(path)
        self._data: dict = {}
        self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                self._data = json.load(f)
        else:
            self._data = {}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def get_all(self) -> dict:
        return dict(self._data)

    def update(self, new_facts: dict):
        """合并新事实，保存"""
        updated = False
        for key, value in new_facts.items():
            if value and str(value).strip():
                self._data[key] = str(value).strip()
                updated = True
        if updated:
            self._save()

    def try_extract_and_update(self, llm, user_input: str, assistant_reply: str) -> bool:
        """
        让 LLM 判断对话中是否有新事实，如果有则更新核心记忆。
        返回是否发生了更新。
        """
        prompt = EXTRACT_FACTS_PROMPT.format(
            user_input=user_input,
            assistant_reply=assistant_reply,
        )
        try:
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip()

            if raw.lower() == "null" or not raw or raw == "{}":
                return False

            # 去掉 markdown 代码块
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            new_facts = json.loads(raw)
            if isinstance(new_facts, dict) and new_facts:
                self.update(new_facts)
                return True
        except Exception:
            pass
        return False
