"""
LLM 提供者工厂
支持 DeepSeek / Qwen / OpenAI / Claude（统一用 ChatOpenAI 兼容接口）
"""
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def get_chat_model(config: dict) -> ChatOpenAI:
    """根据配置返回 LLM 实例"""
    provider = config.get("provider", "deepseek")

    provider_defaults = {
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "api_key_env": "DEEPSEEK_API_KEY",
        },
        "qwen": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key_env": "DASHSCOPE_API_KEY",
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
        },
    }

    defaults = provider_defaults.get(provider, provider_defaults["deepseek"])
    api_key = os.getenv(defaults["api_key_env"]) or config.get("api_key", "")

    if not api_key:
        raise ValueError(
            f"未找到 API Key。请在 .env 中设置 {defaults['api_key_env']}，"
            "或在 config.yaml 的 llm.api_key 中填写。"
        )

    return ChatOpenAI(
        model=config.get("model_name", "deepseek-chat"),
        openai_api_key=api_key,
        openai_api_base=config.get("base_url", defaults["base_url"]),
        temperature=config.get("temperature", 0.85),
        max_tokens=config.get("max_tokens", 512),
    )


def get_embedding_model(config: dict):
    """返回 Embedding 模型实例（本地 BGE 或 OpenAI）"""
    provider = config.get("provider", "openai")

    if provider == "local":
        # 本地 BGE 模型，需要 sentence-transformers
        try:
            from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        except ImportError:
            raise ImportError(
                "使用本地 embedding 需要安装 sentence-transformers：\n"
                "pip install sentence-transformers"
            )
        return HuggingFaceBgeEmbeddings(
            model_name=config.get("model_name", "BAAI/bge-large-zh-v1.5"),
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    else:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")
        return OpenAIEmbeddings(
            model=config.get("openai_model", "text-embedding-3-small"),
            openai_api_key=api_key,
        )
