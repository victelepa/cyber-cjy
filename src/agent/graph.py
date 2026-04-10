"""
Phase 4 LangGraph Agent
替代 simple_agent.py，支持三级记忆系统。
"""
import yaml
from pathlib import Path
from functools import partial
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

from src.llm.provider import get_chat_model
from src.persona.profile import PersonaProfile, MANUAL_FALLBACK_PERSONA
from src.memory.manager import MemoryManager
from src.agent.state import AgentState
from src.agent.nodes import (
    retrieve_memory_node,
    generate_reply_node,
    parse_response_node,
    format_output_node,
    should_send_sticker,
)

load_dotenv()


class CyberGirlfriendAgent:
    """Phase 4 完整 Agent，使用 LangGraph + 三级记忆"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.llm = get_chat_model(self.config["llm"])
        self.memory = MemoryManager(self.config)
        self.persona = self._load_persona()
        self._graph = self._build_graph()

    def _load_persona(self) -> PersonaProfile:
        path = self.config["persona"]["profile_path"]
        use_fallback = self.config["persona"].get("use_manual_fallback", False)
        if Path(path).exists():
            return PersonaProfile.load(path)
        return MANUAL_FALLBACK_PERSONA if use_fallback else PersonaProfile()

    def _build_graph(self) -> object:
        """构建 LangGraph 状态机"""
        workflow = StateGraph(AgentState)

        # 绑定依赖（llm / memory_manager）到节点函数
        workflow.add_node(
            "retrieve_memory",
            partial(retrieve_memory_node, memory_manager=self.memory),
        )
        workflow.add_node(
            "generate_reply",
            partial(generate_reply_node, llm=self.llm),
        )
        workflow.add_node("parse_response", parse_response_node)
        workflow.add_node("format_output", format_output_node)

        # 边
        workflow.set_entry_point("retrieve_memory")
        workflow.add_edge("retrieve_memory", "generate_reply")
        workflow.add_edge("generate_reply", "parse_response")
        workflow.add_edge("parse_response", "format_output")
        workflow.add_edge("format_output", END)

        return workflow.compile()

    def chat(self, user_input: str) -> dict:
        """
        发送一条消息，返回 final_output 字典：
          {
            "text": str,           # 纯文本回复
            "messages": list[str], # 按 --- 分割的多条消息
            "sticker_tag": str | None,
          }
        """
        initial_state: AgentState = {
            "user_input": user_input,
            "messages": self.memory.short_term.get_messages(),
            "retrieved_context": "",
            "few_shot_examples": "",
            "core_memory": self.memory.core.get_all(),
            "persona_section": self.persona.to_system_prompt_section(),
            "response_text": "",
            "sticker_tag": None,
            "final_output": {},
        }

        result = self._graph.invoke(initial_state)
        output = result.get("final_output", {})

        # 更新记忆
        reply_text = output.get("text", "")
        if reply_text:
            self.memory.update(user_input, reply_text, llm=self.llm)

        return output

    def reset(self):
        self.memory.reset_short_term()
