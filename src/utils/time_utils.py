"""时间感知工具 — 根据当前时间生成自然语言描述，注入到 Prompt 中"""
from datetime import datetime


def get_time_context() -> str:
    """返回当前时间的自然语言描述，用于 Prompt 的时间感知"""
    now = datetime.now()
    hour = now.hour
    weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()]

    if 5 <= hour < 9:
        period = "早上"
        mood_hint = "刚起床，可能还有点迷糊"
    elif 9 <= hour < 12:
        period = "上午"
        mood_hint = "精神还不错"
    elif 12 <= hour < 14:
        period = "中午"
        mood_hint = "午饭时间，可能在吃饭或者午休"
    elif 14 <= hour < 18:
        period = "下午"
        mood_hint = "下午时光"
    elif 18 <= hour < 21:
        period = "晚上"
        mood_hint = "下班/放学了，心情比较放松"
    elif 21 <= hour < 24:
        period = "深夜"
        mood_hint = "比较晚了，可能有点困，语气会更温柔慵懒"
    else:
        period = "凌晨"
        mood_hint = "很晚了还没睡，可能有点困"

    return (
        f"现在是{weekday}{period} {now.strftime('%H:%M')}，{mood_hint}。"
    )
