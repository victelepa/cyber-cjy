"""
Phase 1 数据预处理脚本
将 WeFlow / WeChatMsg / PyWxDump 导出的聊天记录转换为标准化 JSONL 格式

使用方式：
  python scripts/preprocess.py --input data/raw/texts/ --output data/processed

支持的导出格式：
  - WeFlow 导出的 JSON（本项目的主要格式，含 isSend / emojiMd5 字段）
  - WeChatMsg 导出的 JSON（messages 数组）
  - WeChatMsg 导出的 CSV

运行后会生成：
  data/processed/messages.jsonl         — 标准化消息流（含 sticker 消息）
  data/processed/conversations/         — 按会话分段的对话
  data/processed/few_shot_pairs.jsonl   — "你说 -> 她回" 对话对
  data/processed/her_sticker_md5s.json  — 她发过的 emoji MD5 列表（用于 Phase 5）
"""
import json
import csv
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Iterator


# ──────────────────────────────────────────────
# 标准消息格式
# ──────────────────────────────────────────────

def make_message(
    sender: str,
    timestamp: str,
    content: str,
    msg_type: str = "text",
    session_id: str = "",
    reply_to: str | None = None,
    sticker_md5: str | None = None,
) -> dict:
    msg = {
        "sender": sender,
        "timestamp": timestamp,
        "type": msg_type,
        "content": content,
        "reply_to": reply_to,
        "session_id": session_id,
    }
    if sticker_md5:
        msg["sticker_md5"] = sticker_md5
    return msg


# ──────────────────────────────────────────────
# WeFlow 格式解析器（主格式）
# ──────────────────────────────────────────────

def parse_weflow_json(filepath: str) -> tuple[list[dict], dict]:
    """
    解析 WeFlow 导出的 JSON 格式。
    返回 (messages, meta)，meta 包含 her_nickname / your_nickname。

    WeFlow 关键字段：
      isSend=0  → 她发的（received）
      isSend=1  → 你发的（sent）
      localType=1   → 文本
      localType=3   → 图片
      localType=47  → 自定义表情包
      localType=34  → 语音（已有文字转录在 content）
      localType=43  → 视频
      localType=10000 → 系统消息
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    session = data.get("session", {})
    raw_msgs = data.get("messages", [])

    # 从 session 推断双方昵称
    # WeFlow session.wxid 是对方的微信ID，nickname 是对方的昵称
    her_nickname = session.get("nickname") or session.get("remark") or "她"
    your_nickname = "我"  # WeFlow 不记录自己的昵称，用"我"代替

    messages = []
    for msg in raw_msgs:
        local_type = msg.get("localType", 1)
        is_send = msg.get("isSend", 0)
        ts_raw = msg.get("createTime") or msg.get("timestamp") or ""
        content = msg.get("content") or ""
        emoji_md5 = msg.get("emojiMd5") or ""

        # 跳过系统消息
        if local_type == 10000:
            continue

        msg_type = _normalize_localtype(local_type)

        # 语音消息：content 是语音转文字结果，当作文本处理
        # 但要标记为 voice 以便后续统计时排除
        if msg_type == "voice" and not content:
            continue

        timestamp = _normalize_timestamp(ts_raw)
        if not timestamp:
            continue

        sender = her_nickname if is_send == 0 else your_nickname

        messages.append(make_message(
            sender=sender,
            timestamp=timestamp,
            content=str(content),
            msg_type=msg_type,
            sticker_md5=emoji_md5 if msg_type == "sticker" else None,
        ))

    meta = {
        "her_nickname": her_nickname,
        "your_nickname": your_nickname,
        "session_wxid": session.get("wxid", ""),
    }
    return messages, meta


def parse_generic_json(filepath: str) -> Iterator[dict]:
    """解析通用 JSON 格式（WeChatMsg 等）"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    messages = data if isinstance(data, list) else data.get("messages", [])

    for msg in messages:
        sender = msg.get("sender") or msg.get("talker") or msg.get("from") or "unknown"
        content = msg.get("content") or msg.get("msg") or msg.get("message") or ""
        ts_raw = msg.get("timestamp") or msg.get("createTime") or msg.get("time") or ""
        msg_type = _normalize_type(msg.get("type", "text"))

        if not content or msg_type == "system":
            continue

        timestamp = _normalize_timestamp(ts_raw)
        if not timestamp:
            continue

        yield make_message(
            sender=sender,
            timestamp=timestamp,
            content=str(content),
            msg_type=msg_type,
        )


def parse_csv(filepath: str) -> Iterator[dict]:
    """解析 CSV 格式的聊天记录（通用）"""
    with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 尝试常见的列名
            sender = (
                row.get("sender") or row.get("talker") or
                row.get("发送者") or row.get("from") or "unknown"
            )
            content = (
                row.get("content") or row.get("message") or
                row.get("内容") or row.get("msg") or ""
            )
            ts_raw = (
                row.get("timestamp") or row.get("time") or
                row.get("时间") or row.get("createTime") or ""
            )
            msg_type = _normalize_type(row.get("type", "text"))

            if not content or msg_type == "system":
                continue

            timestamp = _normalize_timestamp(ts_raw)
            if not timestamp:
                continue

            yield make_message(
                sender=sender,
                timestamp=timestamp,
                content=content,
                msg_type=msg_type,
            )


def _normalize_localtype(local_type: int) -> str:
    """WeFlow localType 数字 → 标准类型字符串"""
    if local_type == 1:
        return "text"
    elif local_type == 3:
        return "image"
    elif local_type == 34:
        return "voice"
    elif local_type == 43:
        return "video"
    elif local_type == 47:
        return "sticker"
    elif local_type == 10000:
        return "system"
    else:
        # 其他类型（分享卡片、位置等）归为 text 或 skip
        return "other"


def _normalize_type(raw_type) -> str:
    """将通用格式的消息类型统一为标准类型"""
    t = str(raw_type).lower()
    if t in ("1", "text", "txt", "文本"):
        return "text"
    elif t in ("3", "image", "img", "图片"):
        return "image"
    elif t in ("43", "video", "视频"):
        return "video"
    elif t in ("34", "voice", "audio", "语音"):
        return "voice"
    elif t in ("47", "sticker", "emoji", "表情"):
        return "sticker"
    elif t in ("10000", "10002", "system", "sys", "系统"):
        return "system"
    else:
        return "text"


def _normalize_timestamp(raw) -> str | None:
    """将各种时间格式统一为 ISO 8601 字符串"""
    if not raw:
        return None
    raw = str(raw).strip()

    # Unix 时间戳（秒）
    if raw.isdigit() and len(raw) == 10:
        return datetime.fromtimestamp(int(raw)).isoformat()

    # Unix 时间戳（毫秒）
    if raw.isdigit() and len(raw) == 13:
        return datetime.fromtimestamp(int(raw) / 1000).isoformat()

    # 常见日期格式
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
    ):
        try:
            return datetime.strptime(raw, fmt).isoformat()
        except ValueError:
            continue

    return None


# ──────────────────────────────────────────────
# 对话分段
# ──────────────────────────────────────────────

def segment_conversations(
    messages: list[dict],
    gap_minutes: int = 30,
) -> list[list[dict]]:
    """按时间间隔将消息流切分为独立对话会话"""
    if not messages:
        return []

    conversations = []
    current = [messages[0]]

    for msg in messages[1:]:
        prev_time = datetime.fromisoformat(current[-1]["timestamp"])
        curr_time = datetime.fromisoformat(msg["timestamp"])
        gap = (curr_time - prev_time).total_seconds() / 60

        if gap > gap_minutes:
            conversations.append(current)
            current = [msg]
        else:
            current.append(msg)

    if current:
        conversations.append(current)

    return conversations


# ──────────────────────────────────────────────
# 对话对提取（few-shot 用）
# ──────────────────────────────────────────────

def extract_few_shot_pairs(
    messages: list[dict],
    her_nickname: str,
    your_nickname: str,
) -> list[dict]:
    """提取"你说 -> 她回"的对话对"""
    pairs = []

    for i, msg in enumerate(messages[:-1]):
        next_msg = messages[i + 1]

        # 你说的 -> 她回的
        if msg["sender"] == your_nickname and next_msg["sender"] == her_nickname:
            if msg["type"] == "text" and next_msg["type"] == "text":
                pairs.append({
                    "input": msg["content"],
                    "output": next_msg["content"],
                    "timestamp": msg["timestamp"],
                })

    return pairs


# ──────────────────────────────────────────────
# 她发的表情包提取（Phase 5 准备）
# ──────────────────────────────────────────────

def extract_her_stickers(
    messages: list[dict],
    her_nickname: str,
    raw_emoji_dir: str,
    output_sticker_dir: str,
) -> list[str]:
    """
    从所有消息中找出她发的 sticker，将对应图片复制到 output_sticker_dir。
    返回成功复制的 MD5 列表。
    """
    raw_dir = Path(raw_emoji_dir)
    out_dir = Path(output_sticker_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 收集她发的所有 emoji MD5
    her_md5s = set()
    for msg in messages:
        if msg["sender"] == her_nickname and msg["type"] == "sticker":
            md5 = msg.get("sticker_md5", "")
            if md5:
                her_md5s.add(md5)

    copied = []
    for md5 in her_md5s:
        # 在 raw emoji 目录里找匹配的文件（任意扩展名）
        for ext in (".png", ".jpg", ".gif", ".webp"):
            src = raw_dir / (md5 + ext)
            if src.exists():
                dst = out_dir / src.name
                shutil.copy2(src, dst)
                copied.append(md5)
                break

    return copied


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="微信聊天记录预处理脚本")
    parser.add_argument("--input", required=True, help="输入文件路径或目录（texts/ 目录或单个 JSON 文件）")
    parser.add_argument("--output", default="data/processed", help="输出目录")
    parser.add_argument("--emoji-dir", default="data/raw/emojis", help="原始 emoji 图片目录")
    parser.add_argument(
        "--format", choices=["weflow", "generic", "csv", "auto"], default="auto",
        help="输入格式（auto 会自动检测 WeFlow 格式）"
    )
    parser.add_argument("--gap", type=int, default=30, help="对话分段间隔（分钟）")
    parser.add_argument("--her", default="", help="覆盖自动检测的她的昵称")
    parser.add_argument("--you", default="", help="覆盖自动检测的你的昵称")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "conversations").mkdir(exist_ok=True)

    # 收集文件
    files = []
    if input_path.is_dir():
        files = list(input_path.glob("*.json")) + list(input_path.glob("*.csv"))
    else:
        files = [input_path]

    all_messages = []
    auto_her = ""
    auto_you = ""

    for filepath in files:
        print(f"解析文件：{filepath.name}")
        try:
            fmt = args.format
            if fmt == "auto":
                # 检测是否为 WeFlow 格式（有 "weflow" key）
                with open(filepath, encoding="utf-8") as f:
                    peek = json.load(f)
                fmt = "weflow" if "weflow" in peek else "generic"

            if fmt == "weflow":
                msgs, meta = parse_weflow_json(str(filepath))
                all_messages.extend(msgs)
                if not auto_her:
                    auto_her = meta["her_nickname"]
                if not auto_you:
                    auto_you = meta["your_nickname"]
                print(f"  WeFlow 格式，{len(msgs)} 条消息，对方昵称：{meta['her_nickname']}")
            elif fmt == "csv":
                msgs = list(parse_csv(str(filepath)))
                all_messages.extend(msgs)
                print(f"  CSV 格式，{len(msgs)} 条消息")
            else:
                msgs = list(parse_generic_json(str(filepath)))
                all_messages.extend(msgs)
                print(f"  通用 JSON 格式，{len(msgs)} 条消息")
        except Exception as e:
            print(f"  警告：解析失败 — {e}")

    if not all_messages:
        print("未找到任何有效消息，请检查输入文件格式。")
        return

    # 昵称：命令行参数优先，否则用自动检测
    her_nickname = args.her or auto_her or "她"
    your_nickname = args.you or auto_you or "我"
    print(f"\n昵称：她={her_nickname}，你={your_nickname}")

    # 过滤：只保留文本和 sticker（语音已转文字当文本处理）
    useful_types = {"text", "voice", "sticker"}
    text_messages = [m for m in all_messages if m["type"] in useful_types]

    # 按时间排序
    text_messages.sort(key=lambda m: m["timestamp"])
    print(f"有效消息：{len(text_messages)} 条（文本 + 语音转写 + 表情包）")

    # 输出标准化 JSONL（含 sticker 消息）
    output_messages = output_path / "messages.jsonl"
    with open(output_messages, "w", encoding="utf-8") as f:
        for msg in text_messages:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
    print(f"已输出：{output_messages}")

    # 对话分段（仅用文本消息，不含 sticker）
    text_only = [m for m in text_messages if m["type"] in ("text", "voice")]
    conversations = segment_conversations(text_only, gap_minutes=args.gap)
    print(f"对话分段：{len(conversations)} 个会话")

    conv_dir = output_path / "conversations"
    for i, conv in enumerate(conversations):
        conv_file = conv_dir / f"conv_{i:04d}.jsonl"
        with open(conv_file, "w", encoding="utf-8") as f:
            for msg in conv:
                msg["session_id"] = f"conv_{i:04d}"
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    # 对话对提取（仅文本）
    pairs = extract_few_shot_pairs(text_only, her_nickname, your_nickname)
    pairs_file = output_path / "few_shot_pairs.jsonl"
    with open(pairs_file, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"对话对：{len(pairs)} 个 → {pairs_file}")

    # 她发的 emoji 提取
    emoji_dir = Path(args.emoji_dir)
    if emoji_dir.exists():
        her_sticker_dir = output_path / "stickers"
        copied = extract_her_stickers(
            text_messages, her_nickname,
            str(emoji_dir), str(her_sticker_dir)
        )
        her_md5s = list({
            m.get("sticker_md5") for m in text_messages
            if m["sender"] == her_nickname and m["type"] == "sticker" and m.get("sticker_md5")
        })
        with open(output_path / "her_sticker_md5s.json", "w", encoding="utf-8") as f:
            json.dump(her_md5s, f, ensure_ascii=False, indent=2)
        print(f"她的表情包：{len(her_md5s)} 个唯一 MD5，已复制 {len(copied)} 个文件 → {her_sticker_dir}")
    else:
        print(f"未找到 emoji 目录（{emoji_dir}），跳过表情包提取")

    # 元数据
    her_sticker_count = sum(
        1 for m in text_messages
        if m["sender"] == her_nickname and m["type"] == "sticker"
    )
    you_sticker_count = sum(
        1 for m in text_messages
        if m["sender"] == your_nickname and m["type"] == "sticker"
    )
    meta = {
        "her_nickname": her_nickname,
        "your_nickname": your_nickname,
        "total_messages": len(text_messages),
        "total_conversations": len(conversations),
        "total_pairs": len(pairs),
        "her_sticker_messages": her_sticker_count,
        "you_sticker_messages": you_sticker_count,
        "date_range": {
            "start": text_messages[0]["timestamp"],
            "end": text_messages[-1]["timestamp"],
        },
    }
    with open(output_path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n预处理完成！")
    print(f"  消息总数: {meta['total_messages']}")
    print(f"  对话会话数: {meta['total_conversations']}")
    print(f"  对话对数: {meta['total_pairs']}")
    print(f"  她发表情包: {her_sticker_count} 次，你发: {you_sticker_count} 次")
    print(f"  时间范围: {meta['date_range']['start'][:10]} ~ {meta['date_range']['end'][:10]}")
    print(f"\n下一步：运行 scripts/analyze_persona.py 生成人设档案")


if __name__ == "__main__":
    main()
