# 赛博cjy 开发日志

> 记录从零到现在的完整开发过程，包括技术决策、踩坑和解决思路。

---

## 项目背景

目标是基于真实微信聊天记录，构建一个能模仿特定人说话风格的对话 Agent。不是泛用型 AI 助手，而是针对一个具体的人——她的口头禅、说话节奏、情绪反应、连发消息的习惯——做深度还原。

整个项目从一份 `PLAN.md` 出发，分阶段实现，每个阶段都能独立运行验证。

---

## 技术选型理由

| 层面 | 选择 | 为什么 |
|------|------|--------|
| LLM 调用 | LangChain `ChatOpenAI` | 一个接口覆盖所有 OpenAI-compatible API（DeepSeek、Qwen、GPT），切换只改 `base_url` + `api_key`，不改代码 |
| Agent 编排 | LangGraph | 状态机模型比单纯的 chain 更适合有分支的流程（要不要发表情包、要不要更新核心记忆），状态透明可调试 |
| LLM 提供商 | DeepSeek-V3 | 中文语境效果好，API 价格极低（约 ¥0.002/轮），接口兼容 OpenAI，一键可换 |
| Embedding | BAAI/bge-small-zh-v1.5 | 专门针对中文训练的 BGE 系列，语义匹配准确，模型小（~130MB），本地运行免费 |
| 向量数据库 | ChromaDB | 本地轻量零运维，LangChain 和 chromadb 原生支持，开发阶段不需要独立服务 |
| 前端 | Gradio | 几十行代码出聊天 UI，适合 MVP 快速验证 |
| 数据格式 | JSONL | 逐行读取，适合大文件（18 万条消息），不需要一次全部加载进内存 |

---

## Phase 1 — 项目骨架 + 数据预处理

### 做了什么

搭建了完整的项目目录结构，包括：
- `config.yaml` — 全局配置，LLM 参数、embedding 参数、记忆窗口大小等都在这里，不硬编码
- `.env.example` / `.env` — API Key 管理，通过 `python-dotenv` 加载
- `requirements.txt` — 依赖清单
- `scripts/export_guide.md` — 给用户看的微信导出图文指引
- `scripts/preprocess.py` — 数据预处理主脚本

### 数据来源

导出工具是 **WeFlow**（微信聊天记录导出工具），导出格式是单个 JSON 文件，路径 `data/raw/texts/私聊_老登.json`。

WeFlow 的 JSON 结构：
```json
{
  "weflow": {"version": "1.0.3"},
  "session": {"wxid": "...", "nickname": "CccJoyyy", "remark": "老登"},
  "messages": [
    {
      "localId": 1,
      "createTime": 1507122769,
      "localType": 1,
      "content": "cjy",
      "isSend": 0,
      "emojiMd5": ""
    }
  ]
}
```

关键字段：
- `isSend`: **0 = 她发的（received），1 = 你发的（sent）**。这个字段解决了"emojis 文件夹里分不清谁发的"问题——完全不需要靠文件名猜，直接用这个字段过滤。
- `localType`: 1=文本，3=图片，34=语音，47=自定义表情包，10000=系统消息
- `emojiMd5`: 表情包文件的 MD5，和 `data/raw/emojis/` 目录下的文件名对应

### 预处理逻辑

1. **格式解析**：专门写了 `parse_weflow_json()` 函数处理 WeFlow 格式，区别于通用 JSON 解析。语音消息（localType=34）已经有语音转文字的 `content`，直接当文本处理，不需要额外的 ASR。

2. **消息标准化**：统一输出为：
   ```json
   {"sender": "CccJoyyy", "timestamp": "2017-10-04T06:12:49", "type": "text", "content": "...", "session_id": ""}
   ```

3. **对话分段**：按时间间隔（默认 30 分钟）切分为独立会话，输出到 `data/processed/conversations/conv_XXXX.jsonl`

4. **对话对提取**：找出"你说 → 她回"的连续消息对，存为 `few_shot_pairs.jsonl`，后续用于 few-shot 示例检索

5. **表情包分离**：通过 `isSend=0` 过滤出她发的表情包 MD5，在 `data/raw/emojis/` 里匹配对应图片文件，复制到 `data/processed/stickers/`

### 数据规模（最终）

| 指标 | 数值 |
|------|------|
| 总消息数 | 188,823 条 |
| 时间跨度 | 2017-10-04 ~ 2026-04-10（约 8.5 年） |
| 对话会话数 | 12,777 个 |
| 对话对（你说→她回） | 32,048 个 |
| 她发的表情包（MD5唯一） | 610 个，其中 373 个有本地文件 |
| 她的平均消息长度 | 7.6 字（极短，86.9% 的消息在 10 字以内） |

---

## Phase 2 — 基础对话（最小可运行版本）

### 做了什么

在没有聊天记录分析结果之前，先让系统跑起来，用手写的 fallback 人设验证整个调用链路。

核心文件：
- `src/llm/provider.py` — LLM 和 Embedding 工厂函数
- `src/persona/profile.py` — `PersonaProfile` Pydantic 模型 + 手写 fallback
- `src/agent/simple_agent.py` — 简单版 Agent（无 LangGraph）
- `src/agent/prompt_builder.py` — System Prompt 组装
- `src/agent/response_parser.py` — 解析 LLM 输出（提取 `[STICKER:tag]` 标记）
- `app.py` — Gradio Web UI

### Prompt 设计

System Prompt 分层注入：
1. **人设区**：昵称、性格特征、口头禅、语气特征、消息长度习惯
2. **核心记忆区**：生日、偏好等关键事实 KV（始终包含）
3. **行为规则**：连发消息用 `---` 分隔、表情包用 `[STICKER:emotion_tag]` 标记
4. **时间感知**：根据当前时间（深夜/早上/下午）生成自然语言描述，影响语气

表情包协议：LLM 在回复末尾加 `[STICKER:happy]` 等标记，前端解析后渲染（Phase 2 先显示文字占位）。

### 踩坑：Gradio 6.0 连续三个 Breaking Changes

安装的 Gradio 版本是 6.0，API 和之前版本有几处不兼容：

**问题1**：`theme` 和 `css` 参数从 `gr.Blocks()` 移到了 `launch()` 里
```python
# 旧写法（报 UserWarning）
with gr.Blocks(theme=gr.themes.Soft(), css="...") as demo:

# 新写法
with gr.Blocks() as demo:
    ...
demo.launch(theme=gr.themes.Soft())
```

**问题2**：`gr.Chatbot()` 移除了 `show_copy_button` 和 `bubble_full_width` 参数，直接报 `TypeError`。

**问题3**：Chatbot 数据格式从元组变成字典：
```python
# 旧格式（报 Error: Data incompatible with messages format）
history.append((user_msg, bot_reply))

# 新格式（Gradio 6.0 默认使用 OpenAI messages 格式）
history.append({"role": "user", "content": user_msg})
history.append({"role": "assistant", "content": bot_reply})
```

**问题4**：`type="messages"` 参数也被移除了，因为新格式已经是默认行为。

### 踩坑：DeepSeek 余额不足

第一次发消息就报错 `Error code: 402 - Insufficient Balance`。这是 API 账户余额问题，不是代码 bug。临时切换到有余额的 OpenAI API 解决。

---

## Phase 3 — 人设自动分析

### 做了什么

写了 `scripts/analyze_persona.py`，用 LLM 从真实聊天记录里提炼结构化人设，替代手写的 fallback。

### 分析策略

直接把 18 万条消息全丢给 LLM 是不现实的（上下文长度限制 + 费用）。采用分批采样策略：

1. 只取她发的文本消息（105,046 条）
2. 按时间均匀分成 6 段（覆盖 2017~2026 年不同时期，捕捉说话风格的变化）
3. 每段随机采样 200 条，共 1,200 条消息
4. 每批分别分析，输出结构化 JSON
5. 再用一次 LLM 调用合并 6 批结果，生成综合人设

分析温度设为 0.3（低温度更稳定，适合分析任务）。用 `with_structured_output` 的方式要求输出 JSON，并做了 markdown 代码块的清洗（LLM 有时会加上 ` ```json ` 包裹）。

另外单独计算了不依赖 LLM 的统计特征：平均消息长度、短/中/长消息比例、连发消息比例（相邻消息时间差 < 30 秒）。

### 生成的人设档案

```json
{
  "nickname": "CccJoyyy",
  "avg_msg_length": 7.6,
  "multi_msg_tendency": true,
  "personality_traits": ["情感丰富且直接", "敏感在意细节", "会撒娇有依赖感", "分享欲强", "偶尔抱怨或暴躁", "有幽默感", "会反思关系", "需要陪伴"],
  "speech_habits": ["呜呜", "嘿嘿", "哈哈", "嗯嗯", "草/卧槽/tmd", "yysy/zyz/okk/dbqdbq", "捏/嘛/呀/啦"],
  "tone_markers": ["频繁使用语气词和拟声词", "常用问号（？/？？）表达疑问或质问", "使用省略号（。。/。。。）表达无奈", "中英文夹杂", "波浪线（～）表达轻松开心"],
  "mbti_guess": "ENFP",
  "pet_names": ["zyz", "宝", "宝宝"],
  "conflict_style": "直接质问或表达不满，有时冷战，但会反思、道歉、要求沟通，情绪过后倾向和解",
  "emoji_style": "混合微信经典表情和基础emoji，频率不高但用在关键情绪点"
}
```

`PersonaProfile.load()` 加了过滤逻辑：只保留模型已知的字段，忽略 LLM 可能输出的未知 key，避免 Pydantic 报错。

---

## Phase 4 — 三级记忆系统 + LangGraph Agent

### 架构设计

```
用户输入
  │
  ▼
[短期记忆]  最近 20 轮对话，直接进 Prompt
[长期记忆]  ChromaDB 向量检索，返回相关历史片段 + few-shot 真实回复示例
[核心记忆]  关键事实 KV JSON，始终在 System Prompt 里
  │
  ▼
LangGraph 状态机
  retrieve_memory → generate_reply → parse_response → format_output
  │
  ▼（每轮结束后异步更新）
  短期记忆 += 本轮对话
  长期记忆 += 本轮对话（向量化存储）
  每 3 轮：LLM 判断是否有新关键事实 → 更新核心记忆
```

### 向量索引构建

`scripts/build_index.py` 构建两个 ChromaDB 集合：

- **`few_shot_pairs`**：32,048 个对话对，以"你说的话"为 document（检索 key），"她的回复"存在 metadata 里。查询时用用户当前输入去检索相似的历史场景，把她当时的真实回复注入 Prompt 作为 few-shot 示例。

- **`conversations`**：12,777 个会话，按每 8 条消息滑动切分（overlap=2），共 36,015 个 chunk。用于检索相关历史对话上下文。

### 踩坑：英文 Embedding 模型对中文无效

**第一次构建**用的是 ChromaDB 默认的 `all-MiniLM-L6-v2`（因为当时 `sentence_transformers` 未安装，自动降级）。这是一个纯英文模型，对中文文本的语义理解几乎为零。

实测验证：用"我们去旅游"查询，返回的结果是完全不相关的句子（"怎么这么说"、"你感觉还好吗"）。

**解决方案**：
1. 安装 `sentence_transformers`
2. 切换为 `BAAI/bge-small-zh-v1.5`（专为中文训练，~130MB）
3. 重建整个索引
4. 关键：`build_index.py` 和 `LongTermMemory`（运行时检索）必须用**同一个** embedding 模型，否则向量空间不一致，检索完全失效

修复后检索测试："旅游 去哪里玩" → 命中了她真实说过的"你喜欢旅游吗！！！之后再一起去！！"。

重建后索引总大小：337MB（两个集合 + SQLite 索引文件）。

### LangGraph Agent 设计

选用 LangGraph 而非直接链式调用的原因：流程里有条件分支（要不要发表情包、要不要更新核心记忆），状态机模型比 if-else 链更清晰，未来扩展节点也更容易。

每个节点都是纯函数，通过 `functools.partial` 注入 `llm` 和 `memory_manager` 依赖，而不是全局变量。

`app.py` 里做了降级保护：
```python
try:
    from src.agent.graph import CyberGirlfriendAgent
    agent = CyberGirlfriendAgent()   # Phase 4 完整版
except Exception:
    from src.agent.simple_agent import SimpleChatAgent
    agent = SimpleChatAgent()        # 降级到 Phase 2 基础版
```

### 踩坑：脚本里 `ModuleNotFoundError: No module named 'src'`

`scripts/` 目录下的脚本直接 `from src.xxx import ...` 会失败，因为 Python 的工作目录是脚本所在位置。解决方法是在脚本头部手动加入项目根目录：

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

---

## 当前状态（Phase 4 完成）

| 功能 | 状态 |
|------|------|
| 微信数据导出 + 预处理 | ✅ 完成 |
| 基础对话（Gradio UI） | ✅ 完成 |
| LLM 人设分析 | ✅ 完成，生成 persona.json |
| 短期记忆（滑动窗口） | ✅ 完成 |
| 长期记忆（ChromaDB 中文检索） | ✅ 完成，bge-small-zh 模型 |
| Few-shot 真实回复检索 | ✅ 完成 |
| 核心记忆自动提取 | ✅ 完成，每 3 轮检查一次 |
| LangGraph 状态机 | ✅ 完成 |
| 表情包系统 | ⬜ Phase 5 待实现 |
| 情绪状态机 | ⬜ Phase 6 待实现 |
| 连发消息模拟 | ⬜ Phase 6 待实现 |

---

## 待解决 / 已知问题

- **RAG 召回精度**：`bge-small` 比英文模型好很多，但中文语义检索本身有局限——聊天消息很短、口语化、没有标准词汇，相似场景不一定能召回。后续可以尝试 `bge-large-zh-v1.5`（效果更好但更慢）。

- **核心记忆误提取**：LLM 判断"是否值得记住"不一定准确，可能把普通闲聊当成关键事实存进去，也可能漏掉真正重要的信息。目前靠人工检查 `data/core_memory.json` 来纠正。

- **连发消息未还原**：她 67% 的连发比例在 Phase 4 里只靠 Prompt 描述（"用 `---` 分隔多条消息"），LLM 执行不稳定，Phase 6 需要专门处理。

- **表情包系统空缺**：目前只输出 `[表情包: happy]` 文字占位，没有真正渲染图片。

---

## 文件结构快照

```
cyber-cjy/
├── PLAN.md                   # 原始设计规划
├── CLAUDE.md                 # Claude Code 项目指引
├── devlog.md                 # 本文件
├── config.yaml               # 全局配置
├── .env                      # API Keys（gitignore）
├── requirements.txt
├── app.py                    # Gradio UI 入口（自动选 Phase 4 或降级 Phase 2）
│
├── scripts/
│   ├── export_guide.md       # 微信导出指引
│   ├── preprocess.py         # 数据预处理（WeFlow JSON → 标准 JSONL）
│   ├── analyze_persona.py    # LLM 人设分析
│   └── build_index.py        # 构建 ChromaDB 向量索引
│
├── src/
│   ├── llm/provider.py       # LLM / Embedding 工厂
│   ├── persona/profile.py    # PersonaProfile 模型 + 手写 fallback
│   ├── memory/
│   │   ├── short_term.py     # 滑动窗口记忆
│   │   ├── long_term.py      # ChromaDB 向量检索
│   │   ├── core_memory.py    # 关键事实 KV + 自动提取
│   │   └── manager.py        # 三级记忆协调器
│   ├── agent/
│   │   ├── simple_agent.py   # Phase 2 基础版（保留作参考）
│   │   ├── graph.py          # Phase 4 LangGraph Agent
│   │   ├── nodes.py          # LangGraph 节点函数
│   │   ├── state.py          # AgentState TypedDict
│   │   ├── prompt_builder.py # Prompt 组装
│   │   └── response_parser.py# 回复解析（文本 + 表情包标记）
│   └── utils/
│       ├── time_utils.py     # 时间感知（深夜/早上等场景描述）
│       └── text.py           # 文本处理工具
│
└── data/                     # gitignore，包含私人聊天记录
    ├── raw/                  # WeFlow 原始导出
    ├── processed/            # 预处理结果
    │   ├── messages.jsonl    # 188,823 条标准化消息
    │   ├── conversations/    # 12,777 个对话会话
    │   ├── few_shot_pairs.jsonl  # 32,048 个对话对
    │   ├── stickers/         # 373 个她发过的表情包图片
    │   └── persona.json      # LLM 生成的人设档案
    ├── chroma_db/            # 337MB 向量索引
    └── core_memory.json      # 核心记忆（可手动编辑）
```
