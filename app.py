"""
赛博cjy Agent — Gradio Web UI 入口
运行：python app.py
然后在浏览器访问 http://localhost:7860
"""
import os
import time
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# 延迟加载 Agent，避免未配置时崩溃
agent = None


def get_agent():
    global agent
    if agent is None:
        # 优先使用 LangGraph Agent（有长期记忆 + 情绪状态机）
        # 若 ChromaDB 索引未建立，自动降级到 simple_agent
        try:
            from src.agent.graph import CyberGirlfriendAgent
            agent = CyberGirlfriendAgent()
        except Exception as e:
            print(f"[警告] LangGraph Agent 加载失败（{e}），降级到基础版")
            from src.agent.simple_agent import SimpleChatAgent
            agent = SimpleChatAgent()
    return agent


def chat(user_message: str, history: list, current_emotion: str):
    """
    Gradio 聊天回调 — generator 版本（Phase 6）。

    连发消息逐条插入气泡，消息间有短暂延迟，模拟真实打字节奏。
    Yields: (msg_input_value, chatbot_history, emotion_markdown)
    """
    if not user_message.strip():
        yield "", history, current_emotion
        return

    # 先把用户消息加入历史
    history = history + [{"role": "user", "content": user_message}]
    yield "", history, current_emotion

    try:
        ag = get_agent()
        result = ag.chat(user_message)

        # LangGraph agent 返回 dict，simple_agent 返回 ParsedResponse
        if isinstance(result, dict):
            messages = result.get("messages") or [result.get("text", "")]
            sticker_path = result.get("sticker_path")
            emotion = result.get("emotion_state", "平静")
        else:
            messages = result.messages
            sticker_path = None
            emotion = "平静"

        emotion_md = f"情绪：**{emotion}**"

        # Phase 6: 连发消息分气泡，逐条加入并 yield（模拟打字延迟）
        for i, msg in enumerate(messages):
            if not msg.strip():
                continue
            history = history + [{"role": "assistant", "content": msg.strip()}]
            yield "", history, emotion_md
            # 消息之间加短暂停顿（最后一条消息后不停顿）
            if i < len(messages) - 1:
                time.sleep(0.6)

        # 表情包作为独立消息插入对话流
        if sticker_path and Path(sticker_path).exists():
            time.sleep(0.3)
            history = history + [{"role": "assistant", "content": {"path": sticker_path}}]
            yield "", history, emotion_md

    except Exception as e:
        error_msg = f"[错误] {str(e)}"
        history = history + [{"role": "assistant", "content": error_msg}]
        yield "", history, current_emotion


def reset_chat():
    """清空对话历史"""
    global agent
    if agent is not None:
        agent.reset()
    return []


def check_config() -> str:
    """检查配置是否就绪"""
    issues = []

    if not Path("config.yaml").exists():
        issues.append("config.yaml 不存在")

    if not Path(".env").exists():
        issues.append(".env 不存在（请复制 .env.example 并填写 API Key）")
    else:
        load_dotenv()
        has_key = (
            os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
        )
        if not has_key:
            issues.append("未找到有效的 API Key（检查 .env 文件）")

    if issues:
        return "配置问题：\n" + "\n".join(f"- {i}" for i in issues)
    return "配置就绪"


# 构建 Gradio UI
with gr.Blocks(title="赛博cjy Agent") as demo:
    gr.Markdown(
        """
        # 赛博cjy Agent
        基于真实聊天记录训练的对话助手
        """
    )

    config_status = check_config()
    if config_status != "配置就绪":
        gr.Warning(config_status)

    with gr.Column():
        chatbot = gr.Chatbot(
            label="对话",
            height=500,
            group_consecutive_messages=False,  # Phase 6: 连发消息各自独立气泡
        )

        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="说点什么...",
                label="",
                scale=9,
                lines=1,
                max_lines=4,
            )
            send_btn = gr.Button("发送", scale=1, variant="primary")

        with gr.Row():
            reset_btn = gr.Button("清空对话", variant="secondary", size="sm")
            emotion_display = gr.Markdown("情绪：平静", elem_id="emotion-status")

    # 事件绑定
    send_btn.click(
        fn=chat,
        inputs=[msg_input, chatbot, emotion_display],
        outputs=[msg_input, chatbot, emotion_display],
    )
    msg_input.submit(
        fn=chat,
        inputs=[msg_input, chatbot, emotion_display],
        outputs=[msg_input, chatbot, emotion_display],
    )
    reset_btn.click(
        fn=reset_chat,
        outputs=[chatbot],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
        allowed_paths=["data/processed/stickers"],
    )
