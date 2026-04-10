"""
Phase 2 简单版 Agent（不含 LangGraph）
直接用 LangChain 的 ChatOpenAI + 对话历史实现基础聊天
Phase 4 升级为 LangGraph 版本后此文件保留作对比
"""
import yaml
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.llm.provider import get_chat_model
from src.persona.profile import PersonaProfile, MANUAL_FALLBACK_PERSONA
from src.agent.prompt_builder import build_system_prompt
from src.agent.response_parser import parse_response, ParsedResponse
from src.utils.time_utils import get_time_context

load_dotenv()


class SimpleChatAgent:
    """Phase 2 基础对话 Agent"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.llm = get_chat_model(self.config["llm"])
        self.chat_history: list = []  # LangChain message 列表
        self.core_memory: dict = self._load_core_memory()
        self.persona: PersonaProfile = self._load_persona()

    def _load_persona(self) -> PersonaProfile:
        profile_path = self.config["persona"]["profile_path"]
        use_fallback = self.config["persona"].get("use_manual_fallback", True)

        if Path(profile_path).exists():
            return PersonaProfile.load(profile_path)
        elif use_fallback:
            return MANUAL_FALLBACK_PERSONA
        else:
            return PersonaProfile()

    def _load_core_memory(self) -> dict:
        path = Path(self.config["memory"]["core_memory_path"])
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _build_messages(self, user_input: str) -> list:
        """组装发给 LLM 的消息列表"""
        system_prompt = build_system_prompt(
            persona_section=self.persona.to_system_prompt_section(),
            core_memory=self.core_memory,
            time_context=get_time_context(),
        )

        messages = [SystemMessage(content=system_prompt)]
        messages.extend(self.chat_history)
        messages.append(HumanMessage(content=user_input))
        return messages

    def chat(self, user_input: str) -> ParsedResponse:
        """发送一条消息，返回解析后的回复"""
        messages = self._build_messages(user_input)
        response = self.llm.invoke(messages)
        raw_text = response.content

        # 更新对话历史（短期记忆）
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=raw_text))

        # 保持最近 N 轮（N*2 条消息）
        window = self.config["memory"]["short_term_window"]
        if len(self.chat_history) > window * 2:
            self.chat_history = self.chat_history[-(window * 2):]

        return parse_response(raw_text)

    def reset(self):
        """清空对话历史"""
        self.chat_history = []
