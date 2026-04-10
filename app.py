"""
赛博cjy Agent — Gradio Web UI 入口
运行：python app.py
然后在浏览器访问 http://localhost:7860
"""
import os
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# 延迟加载 Agent，避免未配置时崩溃
agent = None


def get_agent():
    global agent
    if agent is None:
        # Phase 4: 优先使用 LangGraph Agent（有长期记忆）
        # 若 ChromaDB 索引未建立，自动降级到 simple_agent
        try:
            from src.agent.graph import CyberGirlfriendAgent
            agent = CyberGirlfriendAgent()
        except Exception as e:
            print(f"[警告] LangGraph Agent 加载失败（{e}），降级到基础版")
            from src.agent.simple_agent import SimpleChatAgent
            agent = SimpleChatAgent()
    return agent


def chat(user_message: str, history: list) -> tuple[str, list]:
    """Gradio 聊天回调（Gradio 6.0 messages 格式）"""
    if not user_message.strip():
        return "", history

    history = history + [{"role": "user", "content": user_message}]

    try:
        ag = get_agent()
        result = ag.chat(user_message)

        # LangGraph agent 返回 dict，simple_agent 返回 ParsedResponse
        if isinstance(result, dict):
            reply = "\n".join(result.get("messages", [result.get("text", "")]))
            sticker_tag = result.get("sticker_tag")
        else:
            reply = "\n".join(result.messages)
            sticker_tag = result.sticker_tag

        if sticker_tag:
            reply += f"\n[表情包: {sticker_tag}]"

        history = history + [{"role": "assistant", "content": reply}]
        return "", history

    except Exception as e:
        error_msg = f"[错误] {str(e)}"
        history = history + [{"role": "assistant", "content": error_msg}]
        return "", history


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
            gr.Markdown("*Phase 4 — 三级记忆 + LangGraph*")

    # 事件绑定
    send_btn.click(
        fn=chat,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot],
    )
    msg_input.submit(
        fn=chat,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot],
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
    )
