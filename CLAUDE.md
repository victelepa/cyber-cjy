# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

"赛博cjy Agent" — a conversational AI agent that mimics a specific person's WeChat chat style, built on LangChain + LangGraph + ChromaDB + Gradio. The full implementation plan is in `PLAN.md`.

## Development Commands

```bash
# Setup
cp .env.example .env          # then fill in API keys
pip install -r requirements.txt

# Run the chat UI
python app.py                  # opens http://localhost:7860

# Data pipeline (Phase 1)
python scripts/preprocess.py --input data/raw/ --her "her_nickname" --you "your_nickname"

# Persona analysis (Phase 3, after preprocess)
python scripts/analyze_persona.py   # not yet implemented

# Build vector index (Phase 4)
python scripts/build_index.py       # not yet implemented
```

## Architecture

### Implementation Phases
The project is built incrementally — each phase is independently runnable:
- **Phase 1**: Data preprocessing (`scripts/preprocess.py`) — normalizes WeChat exports to JSONL
- **Phase 2**: Basic chat agent (`src/agent/simple_agent.py` + `app.py`) — works right now, uses manual persona fallback
- **Phase 3**: Persona analysis (`scripts/analyze_persona.py`) — LLM extracts structured persona from chat logs
- **Phase 4**: Memory system (`src/memory/`) + LangGraph agent (`src/agent/graph.py`) — not yet implemented
- **Phase 5**: Sticker system (`src/sticker/`) — not yet implemented
- **Phase 6**: Polish (few-shot retrieval, emotion state machine, multi-message simulation)

### Configuration (`config.yaml`)
All runtime behavior is controlled here: LLM provider/model, embedding provider, memory window sizes, sticker send probability, preprocessing gap. The `persona.use_manual_fallback: true` flag controls whether to use the hand-written fallback persona (`MANUAL_FALLBACK_PERSONA` in `src/persona/profile.py`) when `data/processed/persona.json` doesn't exist yet.

### LLM Provider (`src/llm/provider.py`)
Single factory for all LLM backends via LangChain's `ChatOpenAI`. Switching providers (DeepSeek → Qwen → OpenAI) only requires changing `config.yaml` + the corresponding env var. Embedding similarly abstracted: `local` uses `HuggingFaceBgeEmbeddings` (BAAI/bge-large-zh-v1.5), `openai` uses `OpenAIEmbeddings`.

### Persona System (`src/persona/profile.py`)
`PersonaProfile` is a Pydantic model with `.to_system_prompt_section()` that renders persona fields into natural Chinese for injection into the system prompt. `MANUAL_FALLBACK_PERSONA` is the hand-written baseline used before `analyze_persona.py` runs.

### Agent Flow (Phase 2, `src/agent/simple_agent.py`)
`SimpleChatAgent` maintains a sliding window of `HumanMessage`/`AIMessage` objects in memory. Each call to `.chat()` rebuilds the full message list: `[SystemMessage] + history + [HumanMessage]`. The system prompt is reassembled each turn (persona section + core memory + time context).

### Response Format
LLM responses may contain `[STICKER:emotion_tag]` markers (e.g. `[STICKER:happy]`). Multi-message simulation uses `---` as a delimiter within a single response. `src/agent/response_parser.py` handles both.

### Prompt Assembly (`src/agent/prompt_builder.py`)
System prompt = persona section + core memory KV + behavior rules + time context. Future phases will add: retrieved history (ChromaDB), conversation summary, and few-shot pairs.

### Data Flow
```
data/raw/               → scripts/preprocess.py →
data/processed/
  messages.jsonl        (all messages, normalized)
  conversations/        (time-segmented sessions)
  few_shot_pairs.jsonl  (her_reply given your_input)
  persona.json          (from analyze_persona.py)
data/core_memory.json   (hand-editable KV facts, always in system prompt)
data/chroma_db/         (vector store, built by build_index.py)
```

## Key Conventions

- `data/` is gitignored — contains private chat history. Never commit it.
- `core_memory.json` is the only persistent state in Phase 2 — edit it directly to inject known facts (birthdays, anniversaries, preferences).
- When Phase 4 LangGraph agent is implemented, `simple_agent.py` is retained as a reference; the new entry point will be `src/agent/graph.py`.
- Agent responses with `sticker_tag != None` should eventually render a real image; Phase 2 just appends `[表情包: tag]` as text.
