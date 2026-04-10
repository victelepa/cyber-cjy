"""
表情包库管理器
负责加载/保存 sticker_index.json，提供按情绪标签查询的接口。

索引格式:
{
  "stickers": [
    {
      "id": "sticker_001",
      "filename": "sticker_001.gif",
      "emotion_tags": ["happy", "excited", "laugh"],
      "scene_tags": ["joke", "celebration"],
      "description": "一只小猫开心地跳舞"
    },
    ...
  ]
}
"""
import json
import random
from pathlib import Path


class StickerManager:
    def __init__(self, library_path: str, index_path: str):
        self.library_path = Path(library_path).resolve()
        self.index_path = Path(index_path).resolve()
        self._index: list[dict] = []
        self._load()

    # ── 加载 / 保存 ──────────────────────────────────────────────

    def _load(self):
        if self.index_path.exists():
            with open(self.index_path, encoding="utf-8") as f:
                data = json.load(f)
            self._index = data.get("stickers", [])
        else:
            self._index = []

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump({"stickers": self._index}, f, ensure_ascii=False, indent=2)

    # ── 查询 ──────────────────────────────────────────────────────

    def get_by_emotion(self, emotion_tag: str) -> list[dict]:
        """
        按情绪标签查询匹配的表情包。
        先做精确匹配，再做前缀/子串模糊匹配，兜底返回全部。
        """
        tag = emotion_tag.lower().strip()

        # 1. 精确匹配
        exact = [s for s in self._index if tag in [t.lower() for t in s.get("emotion_tags", [])]]
        if exact:
            return exact

        # 2. 子串模糊匹配（如 "joyful" 匹配 "joy"）
        fuzzy = [
            s for s in self._index
            if any(tag in t.lower() or t.lower() in tag for t in s.get("emotion_tags", []))
        ]
        if fuzzy:
            return fuzzy

        # 3. 兜底: 返回全部（保证一定能拿到东西）
        return self._index

    def get_sticker_path(self, filename: str) -> str | None:
        """返回表情包文件的绝对路径（文件不存在则返回 None）"""
        path = self.library_path / filename
        return str(path) if path.exists() else None

    def pick_random(self, emotion_tag: str) -> str | None:
        """
        按情绪标签随机挑一个表情包，返回其文件路径。
        如果库为空或文件不存在，返回 None。
        """
        candidates = self.get_by_emotion(emotion_tag)
        if not candidates:
            return None

        random.shuffle(candidates)
        for sticker in candidates:
            path = self.get_sticker_path(sticker["filename"])
            if path:
                return path
        return None

    # ── 管理 ──────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        return len(self._index)

    def add_sticker(self, filename: str, emotion_tags: list[str],
                    scene_tags: list[str] = None, description: str = "") -> dict:
        """向索引中添加一个表情包条目"""
        sticker_id = f"sticker_{len(self._index) + 1:04d}"
        entry = {
            "id": sticker_id,
            "filename": filename,
            "emotion_tags": emotion_tags,
            "scene_tags": scene_tags or [],
            "description": description,
        }
        self._index.append(entry)
        return entry

    def all_stickers(self) -> list[dict]:
        return list(self._index)

    def is_empty(self) -> bool:
        return len(self._index) == 0
