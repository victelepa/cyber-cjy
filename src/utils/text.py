"""文本处理工具"""
import re


def clean_wechat_text(text: str) -> str:
    """清洗微信消息文本中的特殊格式"""
    # 移除微信表情文字标记（如 [微笑]、[捂脸]）
    text = re.sub(r'\[[\u4e00-\u9fa5a-zA-Z0-9]+\]', '', text)
    # 移除多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """简单文本分块，用于向量化前的预处理"""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def format_conversation_for_embedding(messages: list[dict]) -> str:
    """将对话列表格式化为单个文本字符串，用于 embedding"""
    lines = []
    for msg in messages:
        sender = msg.get("sender", "unknown")
        content = msg.get("content", "")
        if content:
            lines.append(f"{sender}: {content}")
    return "\n".join(lines)
