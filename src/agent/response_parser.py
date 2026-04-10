"""回复解析器 — 从 LLM 输出中提取文本和表情包指令"""
import re
from dataclasses import dataclass


@dataclass
class ParsedResponse:
    text: str                    # 纯文本部分（去掉表情包标记后）
    sticker_tag: str | None      # 情绪标签，如 "happy" / "shy"，None 表示不发表情包
    messages: list[str]          # 按 "---" 分割后的多条消息列表


def parse_response(raw: str) -> ParsedResponse:
    """解析 LLM 的原始输出"""
    # 提取表情包标记
    sticker_match = re.search(r'\[STICKER:([a-zA-Z_]+)\]', raw)
    sticker_tag = sticker_match.group(1) if sticker_match else None

    # 移除表情包标记，得到纯文本
    text = re.sub(r'\[STICKER:[a-zA-Z_]+\]', '', raw).strip()

    # 先按 "---" 分割，再按换行分割
    # LLM 有时用 ---，有时直接换行，两种都要处理
    raw_parts = text.split("---")
    messages = []
    for part in raw_parts:
        lines = [l.strip() for l in part.split("\n") if l.strip()]
        messages.extend(lines)
    if not messages:
        messages = [text]

    return ParsedResponse(
        text=text,
        sticker_tag=sticker_tag,
        messages=messages,
    )
