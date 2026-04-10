"""
表情包标注脚本 (Phase 5)
扫描 data/processed/stickers/ 目录，为每个图片生成情绪/场景标签，
输出 data/processed/sticker_index.json。

用法:
    # 默认：从聊天上下文关键词推断情绪（不需要额外 API，准确率好）
    python scripts/annotate_stickers.py

    # 用文本 LLM 理解上下文（更准，消耗少量 API token）
    python scripts/annotate_stickers.py --llm-context

    # 用视觉模型直接看图打标签（最准，需要支持 vision 的模型）
    python scripts/annotate_stickers.py --vision

    # 只处理尚未在索引中的新文件
    python scripts/annotate_stickers.py --incremental
"""
import argparse
import base64
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.provider import get_chat_model
from src.sticker.manager import StickerManager

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

# ── 中文聊天情绪关键词（上下文匹配用）────────────────────────────────────

CONTEXT_EMOTION_KEYWORDS: dict[str, list[str]] = {
    "happy": [
        "哈哈", "嘿嘿", "开心", "好耶", "太好了", "棒棒", "耶", "好哦",
        "嘻嘻", "好高兴", "好开心", "棒", "赞", "好的", "好哇",
    ],
    "laugh": [
        "哈哈哈", "笑死", "笑了", "笑哭", "哈哈哈哈", "哈哈哈哈哈",
        "好笑", "lol", "哈哈哈哈哈哈", "笑死我了",
    ],
    "sad": [
        "呜呜", "哭了", "难过", "伤心", "呜", "qaq", "QAQ",
        "555", "心疼", "难受", "泪目", "好难过",
    ],
    "shy": [
        "害羞", "脸红", "嘤嘤", "不要嘛", "羞羞", "羞",
        "好害羞", "啊啊啊啊", "不要", "嗯嗯嗯",
    ],
    "angry": [
        "气死", "烦死", "讨厌", "滚", "生气", "太气了",
        "啊啊啊", "怒了", "气气气", "哼", "凶你",
    ],
    "love": [
        "爱你", "喜欢你", "么么", "亲亲", "么么哒", "❤",
        "爱爱", "好喜欢", "亲", "我喜欢你", "想你",
    ],
    "tired": [
        "累了", "困了", "睡觉", "好累", "懒", "困困",
        "不想动", "好困", "疲惫", "去睡了",
    ],
    "excited": [
        "哇哇哇", "太棒了", "好期待", "激动", "兴奋",
        "wow", "！！！", "哇哦", "好厉害", "真的假的",
    ],
    "cute": [
        "萌", "可爱", "好萌", "嗯嗯", "嘤", "好可爱",
        "萌死了", "太萌了",
    ],
    "surprise": [
        "啊", "什么", "真的假的", "震惊", "天哪", "我的天",
        "不会吧", "卧槽", "哇塞", "吓我一跳",
    ],
    "playful": [
        "嘿嘿嘿", "哼哼", "嘻嘻嘻", "坏笑", "捂嘴",
        "偷笑", "坏坏", "小坏蛋",
    ],
}


def infer_emotion_from_context(context_texts: list[str]) -> list[str]:
    """
    根据周围对话文本用关键词匹配推断情绪标签。
    context_texts: 表情包发送前后几条消息的文本列表。
    """
    combined = " ".join(context_texts)
    scores: dict[str, int] = defaultdict(int)

    for emotion, keywords in CONTEXT_EMOTION_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in combined.lower():
                scores[emotion] += 1

    if not scores:
        return ["cute"]  # 兜底

    # 取得分最高的 1-2 个标签
    sorted_emotions = sorted(scores.items(), key=lambda x: -x[1])
    top = [e for e, s in sorted_emotions if s >= sorted_emotions[0][1] * 0.6]
    return top[:2]


# ── 聊天记录上下文构建 ────────────────────────────────────────────────────

def build_sticker_context_map(
    messages_path: str,
    her_nickname: str,
    window: int = 3,
) -> dict[str, list[str]]:
    """
    读取 messages.jsonl，为每个她发的 sticker MD5 收集上下文文本。
    window: 取 sticker 前后各 window 条文本消息。
    返回: {md5: [context_text, ...]}
    """
    path = Path(messages_path)
    if not path.exists():
        return {}

    messages = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(json.loads(line))

    # 只保留文本和 sticker 消息
    filtered = [m for m in messages if m["type"] in ("text", "voice", "sticker")]

    context_map: dict[str, list[str]] = defaultdict(list)

    for i, msg in enumerate(filtered):
        if msg["sender"] != her_nickname or msg["type"] != "sticker":
            continue
        md5 = msg.get("sticker_md5", "")
        if not md5:
            continue

        # 取前后各 window 条文本消息
        nearby_texts = []
        # 往前找
        j = i - 1
        found_before = 0
        while j >= 0 and found_before < window:
            if filtered[j]["type"] in ("text", "voice") and filtered[j].get("content"):
                nearby_texts.append(filtered[j]["content"])
                found_before += 1
            j -= 1
        # 往后找
        j = i + 1
        found_after = 0
        while j < len(filtered) and found_after < window:
            if filtered[j]["type"] in ("text", "voice") and filtered[j].get("content"):
                nearby_texts.append(filtered[j]["content"])
                found_after += 1
            j += 1

        context_map[md5].extend(nearby_texts)

    return dict(context_map)


# ── 视觉模型标注 ──────────────────────────────────────────────────────────

def encode_image_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


VISION_PROMPT = """\
请分析这张表情包图片，用 JSON 格式回复，包含以下字段：
- emotion_tags: 情绪标签列表（英文小写）
  可选标签: happy, sad, angry, cute, shy, love, excited, tired, cool, surprise, laugh, bored, worried, playful, smug
- scene_tags: 适用场景标签列表（英文小写，如: joke, comfort, greeting, reaction, flirt）
- description: 一句话中文描述图片内容

只输出 JSON，不要其他内容。例如:
{"emotion_tags": ["happy", "excited"], "scene_tags": ["celebration"], "description": "一只小熊开心地举着气球"}
"""


def annotate_with_vision(image_path: Path, llm) -> dict:
    from langchain_core.messages import HumanMessage

    ext = image_path.suffix.lower()[1:]
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "gif": "image/gif", "webp": "image/webp"}
    mime = mime_map.get(ext, "image/png")

    b64 = encode_image_base64(image_path)
    message = HumanMessage(content=[
        {"type": "text", "text": VISION_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ])
    response = llm.invoke([message])
    raw = response.content.strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    return {}


# ── LLM 文本上下文标注 ────────────────────────────────────────────────────

LLM_CONTEXT_PROMPT = """\
以下是一段微信聊天记录的片段，中间某处她发了一个表情包。
请根据上下文语气判断这个表情包最可能表达的情绪，用 JSON 格式回复：
- emotion_tags: 1-2 个英文情绪标签（从 happy/sad/angry/cute/shy/love/excited/tired/surprise/laugh/playful 中选）
- scene_tags: 0-2 个场景标签（从 joke/comfort/greeting/reaction/flirt/rant 中选）

聊天上下文：
{context}

只输出 JSON，不要其他内容。例如: {{"emotion_tags": ["happy"], "scene_tags": ["greeting"]}}
"""


def annotate_with_llm_context(context_texts: list[str], llm) -> dict:
    from langchain_core.messages import HumanMessage

    if not context_texts:
        return {}

    context_str = "\n".join(f"- {t}" for t in context_texts[:6])
    prompt = LLM_CONTEXT_PROMPT.format(context=context_str)
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    return {}


# ── 主流程 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="为表情包批量打标签")
    parser.add_argument("--llm-context", action="store_true",
                        help="用文本 LLM 理解聊天上下文打标签（更准，消耗少量 token）")
    parser.add_argument("--vision", action="store_true",
                        help="用视觉模型直接看图打标签（最准，需 vision 模型）")
    parser.add_argument("--incremental", action="store_true",
                        help="只处理尚未在索引中的新文件")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--messages", default="data/processed/messages.jsonl",
                        help="消息记录路径（用于上下文推断）")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    library_path = Path(config["sticker"]["library_path"])
    index_path = Path(config["sticker"]["index_path"])
    her_nickname = config["preprocessing"].get("target_nickname", "她")

    if not library_path.exists():
        print(f"[错误] 表情包目录不存在: {library_path}")
        sys.exit(1)

    image_files = sorted([
        f for f in library_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_files:
        print(f"[警告] 未找到图片文件: {library_path}")
        sys.exit(0)

    print(f"找到 {len(image_files)} 张图片")

    manager = StickerManager(str(library_path), str(index_path))

    if args.incremental:
        existing = {s["filename"] for s in manager.all_stickers()}
        image_files = [f for f in image_files if f.name not in existing]
        print(f"增量模式: 需要处理 {len(image_files)} 张新图片")
        if not image_files:
            print("没有新图片，索引已是最新。")
            return

    # 构建上下文映射（MD5 → 周围对话文本）
    print(f"读取聊天上下文: {args.messages}")
    context_map = build_sticker_context_map(args.messages, her_nickname)
    print(f"找到 {len(context_map)} 个有上下文的表情包 MD5")

    # 初始化 LLM（如果需要）
    llm = None
    if args.vision or args.llm_context:
        llm_config = dict(config["llm"])
        if args.vision and "deepseek" in llm_config.get("model_name", ""):
            print("[警告] DeepSeek-chat 不支持图片，自动改用 --llm-context 模式")
            args.vision = False
            args.llm_context = True
        if llm is None:
            llm = get_chat_model(llm_config)

    # 选择标注模式
    if args.vision:
        mode = "视觉模型"
    elif args.llm_context:
        mode = "LLM 文本上下文"
    else:
        mode = "关键词上下文"
    print(f"标注模式: {mode}\n")

    success, no_context = 0, 0
    for i, img_path in enumerate(image_files, 1):
        md5 = img_path.stem  # 文件名就是 MD5
        context_texts = context_map.get(md5, [])
        print(f"[{i}/{len(image_files)}] {img_path.name}", end=" ... ")

        try:
            if args.vision and llm:
                result = annotate_with_vision(img_path, llm)
                emotion_tags = result.get("emotion_tags") or []
                scene_tags = result.get("scene_tags", [])
                description = result.get("description", "")
                # 视觉模型无结果时降级到上下文
                if not emotion_tags:
                    emotion_tags = infer_emotion_from_context(context_texts)
                time.sleep(0.3)

            elif args.llm_context and llm:
                if context_texts:
                    result = annotate_with_llm_context(context_texts, llm)
                    emotion_tags = result.get("emotion_tags") or []
                    scene_tags = result.get("scene_tags", [])
                    description = ""
                    if not emotion_tags:
                        emotion_tags = infer_emotion_from_context(context_texts)
                    time.sleep(0.2)
                else:
                    emotion_tags = ["cute"]
                    scene_tags = []
                    description = ""
                    no_context += 1

            else:
                # 默认：关键词上下文匹配
                emotion_tags = infer_emotion_from_context(context_texts)
                scene_tags = []
                description = ""
                if not context_texts:
                    no_context += 1

            manager.add_sticker(
                filename=img_path.name,
                emotion_tags=emotion_tags,
                scene_tags=scene_tags,
                description=description,
            )
            print(f"标签: {emotion_tags}" + ("" if context_texts else " [无上下文]"))
            success += 1

        except Exception as e:
            print(f"失败: {e}")
            manager.add_sticker(
                filename=img_path.name,
                emotion_tags=infer_emotion_from_context(context_texts) if context_texts else ["cute"],
                scene_tags=[],
                description="",
            )
            success += 1  # 降级成功

    manager.save()
    print(f"\n完成！共处理 {success} 张，其中 {no_context} 张无上下文（标为 cute）")
    print(f"索引已保存到: {index_path}（共 {manager.count} 条）")

    if no_context > 0 and not args.llm_context:
        print(f"\n提示: {no_context} 张表情包没有找到聊天上下文。")
        print("  可以运行 --llm-context 模式让 LLM 来判断，效果更好。")


if __name__ == "__main__":
    main()
