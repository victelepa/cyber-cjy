"""LangGraph 各节点实现"""
from langchain_core.messages import SystemMessage, HumanMessage

from src.agent.prompt_builder import build_system_prompt, build_context_block
from src.agent.response_parser import parse_response
from src.agent.state import AgentState
from src.utils.time_utils import get_time_context
from src.utils.emotion import infer_emotion
from src.sticker.matcher import StickerMatcher


def retrieve_memory_node(state: AgentState, memory_manager) -> dict:
    """节点1: 检索三级记忆"""
    retrieved = memory_manager.retrieve(state["user_input"])
    return {
        "retrieved_context": retrieved["retrieved_context"],
        "few_shot_examples": retrieved["few_shot_examples"],
        "core_memory": retrieved["core_memory"],
    }


def build_prompt_node(state: AgentState) -> dict:
    """节点2: 组装 Prompt（不修改 state，纯计算）"""
    # 实际 prompt 在 generate_reply_node 里拼装，此节点仅透传
    return {}


def generate_reply_node(state: AgentState, llm) -> dict:
    """节点3: 调用 LLM 生成回复"""
    system_prompt = build_system_prompt(
        persona_section=state["persona_section"],
        core_memory=state["core_memory"],
        time_context=get_time_context(),
        emotion_state=state.get("emotion_state", "平静"),
    )

    context_block = build_context_block(
        retrieved_context=state.get("retrieved_context", ""),
        history_summary="",
        few_shot_examples=state.get("few_shot_examples", ""),
    )

    # 组装消息列表：system + 历史 + (可选)上下文 + 用户输入
    messages = [SystemMessage(content=system_prompt)]
    if context_block:
        messages.append(SystemMessage(content=context_block))
    messages.extend(state.get("messages", []))
    messages.append(HumanMessage(content=state["user_input"]))

    response = llm.invoke(messages)
    return {"response_text": response.content}


def parse_response_node(state: AgentState) -> dict:
    """节点4: 解析回复，提取表情包标记"""
    parsed = parse_response(state["response_text"])
    return {
        "response_text": parsed.text,
        "sticker_tag": parsed.sticker_tag,
    }


def match_sticker_node(state: AgentState, sticker_matcher: StickerMatcher) -> dict:
    """节点5: 从表情包库中匹配真实图片文件"""
    sticker_tag = state.get("sticker_tag")
    sticker_path = sticker_matcher.match(sticker_tag)
    return {"sticker_path": sticker_path}


def format_output_node(state: AgentState) -> dict:
    """节点6: 格式化最终输出"""
    parsed = parse_response(state["response_text"])
    return {
        "final_output": {
            "text": state["response_text"],
            "messages": parsed.messages,
            "sticker_tag": state.get("sticker_tag"),
            "sticker_path": state.get("sticker_path"),
        }
    }


def update_emotion_node(state: AgentState) -> dict:
    """节点7（Phase 6）: 根据本轮对话更新情绪状态"""
    new_emotion = infer_emotion(
        user_input=state.get("user_input", ""),
        bot_reply=state.get("response_text", ""),
        current_emotion=state.get("emotion_state", "平静"),
    )
    return {"emotion_state": new_emotion}


# ── 条件边函数 ──

def should_send_sticker(state: AgentState) -> str:
    return "yes" if state.get("sticker_tag") else "no"
