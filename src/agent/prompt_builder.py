"""Prompt 组装器"""
from src.utils.time_utils import get_time_context


SYSTEM_TEMPLATE = """\
你正在扮演一个真实的人，用她真实的说话风格和对方聊天。

## 人设信息
{persona_section}

## 核心记忆（关键事实）
{core_memory_section}

## 行为规则
- 用真实、自然的中文聊天语气，不要用书面语
- 不要说"作为AI"、"我是语言模型"之类的话，你就是她本人
- 消息长度要符合她的习惯，不要突然变得很长很正式
- 如果她喜欢连发，可以在回复里用换行分隔多条消息（用「---」分隔）
- 回复要有情绪、有温度，不要像客服或百科全书
- 如果你决定发表情包，在回复末尾加上 [STICKER:emotion_tag]，emotion_tag 是情绪描述（如 happy/shy/angry/sad/cute）
- 不要每条都发表情包，要符合她真实的发表情包频率

## 时间感知
{time_context}

## 格式示例
用户说：你今天怎么样

正常回复示例：
还好啦～就是有点困
下午一直在划水哈哈哈
你呢你呢

带表情包示例：
啊啊啊好开心！！
[STICKER:happy]
"""

CONTEXT_TEMPLATE = """\
## 相关历史对话片段（供参考，回忆过去聊过的内容）
{retrieved_context}
"""

FEW_SHOT_TEMPLATE = """\
## 参考示例（你过去在类似场景下的真实回复风格）
{few_shot_examples}
"""


def build_system_prompt(
    persona_section: str,
    core_memory: dict,
    time_context: str | None = None,
) -> str:
    """组装 System Prompt"""
    if core_memory:
        # 过滤掉空值和注释字段
        core_lines = [
            f"- {k}: {v}" for k, v in core_memory.items()
            if v and not k.startswith("备注")
        ]
        core_memory_section = "\n".join(core_lines) if core_lines else "（暂无记录）"
    else:
        core_memory_section = "（暂无记录）"

    return SYSTEM_TEMPLATE.format(
        persona_section=persona_section or "（人设档案尚未生成，使用默认风格）",
        core_memory_section=core_memory_section,
        time_context=time_context or get_time_context(),
    )


def build_context_block(
    retrieved_context: str = "",
    history_summary: str = "",
    few_shot_examples: str = "",
) -> str:
    """组装上下文块（历史检索 + few-shot 示例）"""
    parts = []
    if retrieved_context and retrieved_context.strip():
        parts.append(CONTEXT_TEMPLATE.format(retrieved_context=retrieved_context))
    if few_shot_examples and few_shot_examples.strip():
        parts.append(FEW_SHOT_TEMPLATE.format(few_shot_examples=few_shot_examples))
    return "\n".join(parts)
