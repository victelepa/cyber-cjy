"""
表情包匹配器
根据 LLM 输出的 emotion_tag，从表情包库中挑选合适的图片。

使用方式:
    from src.sticker.manager import StickerManager
    from src.sticker.matcher import StickerMatcher

    manager = StickerManager(library_path, index_path)
    matcher = StickerMatcher(manager, send_probability=0.3)
    path = matcher.match("happy")   # 返回文件路径 or None
"""
import random
from src.sticker.manager import StickerManager

# LLM 可能输出的英文情绪标签 → 规范化映射
# （将近义词/变体统一到索引里常见的标签）
EMOTION_ALIASES: dict[str, list[str]] = {
    "happy":    ["happy", "joy", "joyful", "glad", "pleased", "高兴", "开心"],
    "excited":  ["excited", "thrilled", "pumped", "兴奋", "激动"],
    "laugh":    ["laugh", "lol", "haha", "funny", "哈哈", "大笑"],
    "cute":     ["cute", "sweet", "adorable", "萌", "可爱"],
    "shy":      ["shy", "embarrassed", "blush", "害羞", "脸红"],
    "love":     ["love", "heart", "affection", "爱", "心心"],
    "sad":      ["sad", "cry", "crying", "unhappy", "伤心", "哭泣"],
    "angry":    ["angry", "mad", "furious", "生气", "愤怒"],
    "tired":    ["tired", "sleepy", "exhausted", "困", "累"],
    "surprise": ["surprise", "surprised", "shocked", "wow", "惊讶", "震惊"],
    "cool":     ["cool", "awesome", "slick", "帅", "酷"],
    "confused": ["confused", "puzzled", "uncertain", "hmm", "疑惑", "困惑", "???"],
    "playful":  ["playful", "teasing", "naughty", "调皮", "坏笑"],
    "smug":     ["smug", "proud", "得意", "嘿嘿"],
    "worried":  ["worried", "nervous", "anxious", "担心", "紧张"],
    "bored":    ["bored", "whatever", "无聊", "随便"],
}

# 反向映射: 变体 → 规范标签
_ALIAS_REVERSE: dict[str, str] = {}
for canonical, aliases in EMOTION_ALIASES.items():
    for alias in aliases:
        _ALIAS_REVERSE[alias.lower()] = canonical


def normalize_emotion(tag: str) -> str:
    """将 LLM 输出的情绪标签规范化为索引中的标签"""
    return _ALIAS_REVERSE.get(tag.lower(), tag.lower())


class StickerMatcher:
    """
    根据情绪标签从表情包库中随机挑选一个表情包。

    发送频率由 LLM 通过 prompt 控制（prompt 里已说明不要每条都发）。
    到这里说明 LLM 已经决定发，直接找匹配的图片返回路径即可。
    """

    def __init__(self, manager: StickerManager, send_probability: float = 0.3):
        self.manager = manager
        # send_probability 保留字段但不在此处过滤，由 prompt 控制频率

    def match(self, emotion_tag: str | None) -> str | None:
        """
        匹配表情包，返回文件路径或 None。

        Args:
            emotion_tag: LLM 输出的情绪标签（如 "happy"），None 表示不发

        Returns:
            表情包文件的绝对路径，或 None（库为空 / 无匹配文件）
        """
        if not emotion_tag:
            return None

        if self.manager.is_empty():
            return None

        normalized = normalize_emotion(emotion_tag)
        return self.manager.pick_random(normalized)
