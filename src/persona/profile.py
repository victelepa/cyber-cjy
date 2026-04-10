"""人设档案数据结构"""
import json
from pathlib import Path
from pydantic import BaseModel, Field


class PersonaProfile(BaseModel):
    nickname: str = Field(default="", description="她的昵称")
    speech_habits: list[str] = Field(
        default_factory=list,
        description="口头禅、常用语气词，如 '哈哈哈'、'呜呜'、'好耶'"
    )
    emoji_style: str = Field(default="", description="表情使用风格描述")

    mbti_guess: str = Field(default="", description="推测的 MBTI 类型")
    personality_traits: list[str] = Field(
        default_factory=list,
        description="性格特征，如 ['温柔', '偶尔毒舌', '爱撒娇']"
    )

    avg_msg_length: float = Field(default=20.0, description="平均消息长度（字符数）")
    multi_msg_tendency: bool = Field(default=False, description="是否倾向连发多条短消息")
    response_patterns: dict = Field(
        default_factory=dict,
        description="回复模式描述"
    )
    tone_markers: list[str] = Field(
        default_factory=list,
        description="语气标记特征"
    )

    frequent_topics: list[str] = Field(default_factory=list, description="常聊话题")
    emotional_triggers: dict = Field(
        default_factory=dict,
        description="情绪触发点，如 {'开心': ['...'], '难过': ['...']}"
    )

    pet_names: list[str] = Field(default_factory=list, description="对你的称呼/昵称")
    conflict_style: str = Field(default="", description="吵架时的表现风格")
    affection_style: str = Field(default="", description="表达喜欢的方式")
    # analyze_persona.py 输出的额外字段
    affection_expressions: list[str] = Field(default_factory=list, description="表达亲密的方式列表")

    @classmethod
    def load(cls, path: str) -> "PersonaProfile":
        """从 JSON 文件加载人设档案，忽略未知字段"""
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 只保留模型已知的字段，忽略多余字段（避免 LLM 输出未知 key 导致报错）
        known = cls.model_fields.keys()
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def save(self, path: str) -> None:
        """保存人设档案到 JSON 文件"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, ensure_ascii=False, indent=2)

    def to_system_prompt_section(self) -> str:
        """将人设档案转换为 System Prompt 片段"""
        lines = []

        if self.nickname:
            lines.append(f"你的名字叫「{self.nickname}」。")

        if self.personality_traits:
            traits = "、".join(self.personality_traits)
            lines.append(f"你的性格特点：{traits}。")

        if self.speech_habits:
            habits = "、".join(f"「{h}」" for h in self.speech_habits)
            lines.append(f"你的口头禅和常用语气词：{habits}。")

        if self.tone_markers:
            markers = "、".join(self.tone_markers)
            lines.append(f"你的语气特征：{markers}。")

        if self.avg_msg_length > 0:
            length_desc = "短" if self.avg_msg_length < 15 else "中等" if self.avg_msg_length < 40 else "较长"
            lines.append(f"你的消息一般比较{length_desc}（平均约 {int(self.avg_msg_length)} 个字）。")

        if self.multi_msg_tendency:
            lines.append("你喜欢连发多条短消息，而不是把所有内容塞在一条里。")

        if self.frequent_topics:
            topics = "、".join(self.frequent_topics)
            lines.append(f"你喜欢聊的话题：{topics}。")

        # affection_style 或 affection_expressions 任选其一
        affection = self.affection_style or (
            "、".join(self.affection_expressions) if self.affection_expressions else ""
        )
        if affection:
            lines.append(f"你表达亲密和喜欢的方式：{affection}。")

        if self.conflict_style:
            lines.append(f"你不开心时的表现：{self.conflict_style}。")

        if self.pet_names:
            names = "、".join(f"「{n}」" for n in self.pet_names)
            lines.append(f"你对对方的昵称：{names}。")

        if self.mbti_guess:
            lines.append(f"你的性格倾向（MBTI 参考）：{self.mbti_guess}。")

        if self.emoji_style:
            lines.append(f"表情包使用风格：{self.emoji_style}。")

        return "\n".join(lines)


# 手写的基础人设（在 analyze_persona.py 运行之前作为 fallback）
MANUAL_FALLBACK_PERSONA = PersonaProfile(
    nickname="CccJoyyy",
    personality_traits=["温柔体贴", "偶尔撒娇", "有点小傲娇", "情绪细腻"],
    speech_habits=["哈哈哈", "呜呜", "好耶", "嗯嗯", "啊啊啊"],
    tone_markers=["语气温柔", "偶尔用叠词", "喜欢用省略号表达情绪"],
    avg_msg_length=18.0,
    multi_msg_tendency=True,
    frequent_topics=["日常生活", "美食", "心情"],
    affection_style="用语气词和表情包表达，偶尔直接说",
    pet_names=["你", "笨蛋", "哥哥"],
    emoji_style="喜欢发，尤其是开心或者撒娇的时候",
)
