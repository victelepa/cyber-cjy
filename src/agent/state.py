"""LangGraph Agent 状态定义"""
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # 当前用户输入
    user_input: str
    # 对话消息历史（LangGraph 管理，用于内部传递）
    messages: Annotated[list[BaseMessage], add_messages]
    # 从记忆系统检索到的内容
    retrieved_context: str
    few_shot_examples: str
    # 核心记忆和人设快照
    core_memory: dict
    persona_section: str
    # Agent 生成的回复
    response_text: str
    # 表情包决策
    sticker_tag: str | None
    sticker_path: str | None  # 实际图片文件路径（Phase 5）
    # 最终输出
    final_output: dict
