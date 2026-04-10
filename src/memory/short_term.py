"""短期记忆 — 滑动窗口对话历史"""
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


class ShortTermMemory:
    def __init__(self, window: int = 20):
        self.window = window          # 保留最近 N 轮（N*2 条消息）
        self.messages: list[BaseMessage] = []

    def add_exchange(self, user_input: str, assistant_reply: str):
        self.messages.append(HumanMessage(content=user_input))
        self.messages.append(AIMessage(content=assistant_reply))
        # 超出窗口时截断
        max_msgs = self.window * 2
        if len(self.messages) > max_msgs:
            self.messages = self.messages[-max_msgs:]

    def get_messages(self) -> list[BaseMessage]:
        return list(self.messages)

    def reset(self):
        self.messages = []
