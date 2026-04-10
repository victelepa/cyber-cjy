"""
Phase 3 人设分析脚本
从预处理后的聊天记录中，用 LLM 自动提炼结构化人设档案。

使用方式：
  python scripts/analyze_persona.py

流程：
  1. 从 messages.jsonl 中均匀采样，只取"她"发的文本消息
  2. 分批次送给 LLM 分析，每批约 200 条消息
  3. 合并多批结果，生成综合人设
  4. 保存到 data/processed/persona.json

完成后 app.py 会自动加载真实人设，不再用手写 fallback。
"""
import json
import random
import sys
import yaml
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

# 确保项目根目录在 Python 路径中
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()


# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────

MESSAGES_PATH = "data/processed/messages.jsonl"
PAIRS_PATH = "data/processed/few_shot_pairs.jsonl"
OUTPUT_PATH = "data/processed/persona.json"
CONFIG_PATH = "config.yaml"

BATCH_SIZE = 200       # 每批分析的消息数
NUM_BATCHES = 6        # 分析批次（覆盖不同时间段）
PAIRS_SAMPLE = 60      # 用于分析的对话对样本数


# ──────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────

def load_her_messages(her_nickname: str) -> list[dict]:
    """只加载她发的文本消息"""
    messages = []
    with open(MESSAGES_PATH, encoding="utf-8") as f:
        for line in f:
            msg = json.loads(line)
            if msg["sender"] == her_nickname and msg["type"] in ("text", "voice"):
                if msg["content"].strip():
                    messages.append(msg)
    return messages


def load_pairs(sample_n: int) -> list[dict]:
    """加载对话对样本"""
    pairs = []
    with open(PAIRS_PATH, encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    if len(pairs) > sample_n:
        pairs = random.sample(pairs, sample_n)
    return pairs


def sample_by_period(messages: list[dict], n_batches: int, batch_size: int) -> list[list[dict]]:
    """
    按时间段均匀采样，确保覆盖早期、中期、近期的说话风格
    """
    if not messages:
        return []

    # 按时间分段
    chunk_size = len(messages) // n_batches
    batches = []
    for i in range(n_batches):
        start = i * chunk_size
        end = start + chunk_size if i < n_batches - 1 else len(messages)
        segment = messages[start:end]
        # 在每段里随机采样
        sample = random.sample(segment, min(batch_size, len(segment)))
        batches.append(sample)

    return batches


# ──────────────────────────────────────────────
# 统计特征（不依赖 LLM，快速计算）
# ──────────────────────────────────────────────

def compute_stats(her_messages: list[dict], pairs: list[dict]) -> dict:
    """计算可以直接统计的特征，不需要 LLM"""
    contents = [m["content"] for m in her_messages if m["content"]]

    # 平均消息长度
    avg_len = sum(len(c) for c in contents) / len(contents) if contents else 0

    # 消息长度分布
    short = sum(1 for c in contents if len(c) <= 10)
    medium = sum(1 for c in contents if 10 < len(c) <= 30)
    long = sum(1 for c in contents if len(c) > 30)
    total = len(contents)

    # 常用字符/词
    all_text = "".join(contents)
    # 找出常用的语气词模式
    import re
    # 提取重复字符（如 哈哈哈、呜呜呜）
    repeated = re.findall(r'([\u4e00-\u9fa5])\1{1,}', all_text)
    repeated_counts = Counter(repeated)

    # 连发消息检测（通过 few-shot pairs 分析）
    # 统计她一次回复多条的情况（这里用消息间时间差 < 30秒估算）
    consecutive_count = 0
    for i in range(1, len(her_messages)):
        prev = her_messages[i - 1]
        curr = her_messages[i]
        from datetime import datetime
        try:
            t1 = datetime.fromisoformat(prev["timestamp"])
            t2 = datetime.fromisoformat(curr["timestamp"])
            if (t2 - t1).total_seconds() < 30:
                consecutive_count += 1
        except Exception:
            pass

    multi_msg_ratio = consecutive_count / len(her_messages) if her_messages else 0

    return {
        "avg_msg_length": round(avg_len, 1),
        "msg_length_distribution": {
            "short_pct": round(short / total * 100, 1) if total else 0,
            "medium_pct": round(medium / total * 100, 1) if total else 0,
            "long_pct": round(long / total * 100, 1) if total else 0,
        },
        "multi_msg_tendency": multi_msg_ratio > 0.2,
        "multi_msg_ratio": round(multi_msg_ratio, 2),
        "top_repeated_chars": [char for char, _ in repeated_counts.most_common(10)],
        "total_her_messages": len(her_messages),
    }


# ──────────────────────────────────────────────
# LLM 分析
# ──────────────────────────────────────────────

BATCH_ANALYSIS_PROMPT = """\
以下是一个女生在微信上的真实聊天记录片段（只有她发的消息）。

请仔细分析她的说话风格，输出一个 JSON 对象，包含以下字段：

{{
  "speech_habits": ["她的口头禅和常用语气词，列举真实出现的，如 '哈哈哈'、'呜呜'、'好耶'、'嗯嗯' 等"],
  "personality_traits": ["从消息中观察到的性格特征，如 '温柔'、'爱撒娇'、'偶尔毒舌' 等"],
  "tone_markers": ["语气特征描述，如 '喜欢用省略号'、'常用感叹号'、'语气词丰富' 等"],
  "frequent_topics": ["她经常聊的话题类型"],
  "emotional_triggers": {{
    "happy": ["什么让她开心"],
    "sad": ["什么让她难过"],
    "excited": ["什么让她兴奋"]
  }},
  "affection_expressions": ["她表达喜欢/亲密的方式"],
  "conflict_style": "她不开心或生气时的表现（一句话描述）",
  "emoji_style": "她使用表情符号/颜文字的风格（一句话描述）"
}}

聊天记录（{count} 条，时间范围 {start} ~ {end}）：
{messages}

只输出 JSON，不要有其他文字。"""

MERGE_PROMPT = """\
以下是对同一个女生不同时期聊天记录的多次分析结果（JSON 格式）。
请综合这些分析，生成一份最终的、准确的人设档案。

要求：
- speech_habits: 取各批次都出现过的 + 出现频率高的，去重，列出最有代表性的 8-15 个
- personality_traits: 综合所有观察，去重，列出最准确的 5-8 个
- tone_markers: 综合去重，3-6 个
- frequent_topics: 综合去重，5-8 个
- emotional_triggers: 各类别取 2-4 个最有代表性的
- affection_expressions: 去重，取 3-5 个
- conflict_style: 综合多次描述，用一句话总结
- emoji_style: 综合多次描述，用一句话总结
- mbti_guess: 根据性格特征推测 MBTI（如 ENFP、ISFJ 等，给出简短理由）
- pet_names: 她对你（对方）常用的称呼

多次分析结果：
{analyses}

只输出最终的 JSON，字段同上（加上 mbti_guess 和 pet_names），不要有其他文字。"""


def format_messages_for_prompt(batch: list[dict]) -> str:
    lines = []
    for msg in batch:
        ts = msg["timestamp"][:16].replace("T", " ")
        lines.append(f"[{ts}] {msg['content']}")
    return "\n".join(lines)


def analyze_batch(llm, batch: list[dict], batch_idx: int) -> dict | None:
    """分析一批消息，返回解析后的 JSON"""
    if not batch:
        return None

    start = batch[0]["timestamp"][:10]
    end = batch[-1]["timestamp"][:10]

    prompt = BATCH_ANALYSIS_PROMPT.format(
        count=len(batch),
        start=start,
        end=end,
        messages=format_messages_for_prompt(batch),
    )

    print(f"  分析第 {batch_idx + 1} 批（{start} ~ {end}，{len(batch)} 条消息）...")

    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    # 去掉可能的 markdown 代码块
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"    警告：JSON 解析失败 ({e})，跳过此批次")
        return None


def merge_analyses(llm, analyses: list[dict], pairs: list[dict]) -> dict:
    """合并多批次分析结果"""
    # 加入对话对信息，帮助分析称呼
    pairs_text = "\n".join(
        f"你: {p['input'][:50]}\n她: {p['output'][:80]}"
        for p in pairs[:20]
    )

    analyses_json = json.dumps(analyses, ensure_ascii=False, indent=2)

    prompt = MERGE_PROMPT.format(analyses=analyses_json)
    # 附加对话对
    prompt += f"\n\n另外，以下是一些「你说 → 她回」的真实对话对，用于辅助分析称呼和互动风格：\n{pairs_text}"

    print("  合并多批次分析结果...")

    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main():
    # 加载配置
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    her_nickname = config["preprocessing"]["target_nickname"]
    if not her_nickname:
        # 从 meta.json 读取
        meta_path = Path("data/processed/meta.json")
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            her_nickname = meta.get("her_nickname", "")
    if not her_nickname:
        print("错误：未找到她的昵称，请在 config.yaml 的 preprocessing.target_nickname 中填写")
        return

    print(f"分析对象：{her_nickname}")

    # 加载数据
    print("加载聊天记录...")
    her_messages = load_her_messages(her_nickname)
    print(f"  她的文本消息：{len(her_messages)} 条")

    pairs = load_pairs(PAIRS_SAMPLE)
    print(f"  对话对样本：{len(pairs)} 个")

    # 统计特征
    print("计算统计特征...")
    stats = compute_stats(her_messages, pairs)
    print(f"  平均消息长度：{stats['avg_msg_length']:.1f} 字")
    print(f"  连发消息比例：{stats['multi_msg_ratio']:.1%}")

    # 初始化 LLM（分析用更大 max_tokens）
    llm_config = config["llm"].copy()
    llm_config["max_tokens"] = 2000
    llm_config["temperature"] = 0.3   # 分析任务用低温度，更稳定

    from src.llm.provider import get_chat_model
    llm = get_chat_model(llm_config)

    # 分批分析
    print(f"\n开始 LLM 分析（共 {NUM_BATCHES} 批）...")
    batches = sample_by_period(her_messages, NUM_BATCHES, BATCH_SIZE)

    analyses = []
    for i, batch in enumerate(batches):
        result = analyze_batch(llm, batch, i)
        if result:
            analyses.append(result)

    if not analyses:
        print("错误：所有批次分析均失败，请检查 API 配置")
        return

    print(f"  成功分析 {len(analyses)}/{NUM_BATCHES} 批次")

    # 合并分析结果
    print("\n合并分析结果...")
    merged = merge_analyses(llm, analyses, pairs)

    # 与统计特征合并
    final_persona = {
        "nickname": her_nickname,
        "avg_msg_length": stats["avg_msg_length"],
        "multi_msg_tendency": stats["multi_msg_tendency"],
        "msg_length_distribution": stats["msg_length_distribution"],
        **merged,
    }

    # 保存
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_persona, f, ensure_ascii=False, indent=2)

    print(f"\n人设档案已保存到 {OUTPUT_PATH}")
    print("\n生成的人设摘要：")
    print(f"  性格：{', '.join(final_persona.get('personality_traits', []))}")
    print(f"  口头禅：{', '.join(final_persona.get('speech_habits', [])[:5])}")
    print(f"  MBTI 推测：{final_persona.get('mbti_guess', '未知')}")
    print(f"  称呼你：{', '.join(final_persona.get('pet_names', []))}")
    print(f"\n下一步：重启 app.py，将自动加载真实人设（不再用手写 fallback）")


if __name__ == "__main__":
    main()
