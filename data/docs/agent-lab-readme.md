# agent-lab

[![CI](https://github.com/westsonGoogle/agent-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/westsonGoogle/agent-lab/actions/workflows/ci.yml)

> 第 0 阶段作业：不用任何 Agent 框架，手写一个能跑的 Agent。
> 核心结论先放这里：**Agent = 带工具和消息状态的 while 循环。**

一个命令行 Agent，能：

- 查天气（Open-Meteo，免费无需 Key）
- 读本地文件、按关键词搜索（沙箱限制在 `ALLOWED_ROOT` 内）
- 算数学（AST 白名单求值，不用 `eval`）

同时带一个 FastAPI + SSE 的 HTTP 接口，和一套不联网的 pytest。

---

## 1. 五分钟跑起来

```bash
# 安装(uv 会自动建 .venv 并装好依赖,不要再用 pip + venv)
uv sync

# 配置模型(任何 OpenAI 兼容端点)
cp .env.example .env
# 编辑 .env:LLM_BASE_URL / LLM_API_KEY / LLM_MODEL

# 跑
uv run agent
```

交互示例：

```
你 > 北京现在天气怎么样?顺便算一下 (23*17+9)/4

助手 > 北京现在 18.4°C,多云,体感 17.9°C……
        (23*17+9)/4 = 100.0
```

一次性提问（方便脚本/CI）：

```bash
uv run agent --once "读一下 pyproject.toml 前 10 行,告诉我项目名"
```

### 环境变量

| 变量 | 必填 | 说明 |
| --- | --- | --- |
| `LLM_BASE_URL` | 是 | 到 `/v1` 为止，代码会拼 `/chat/completions` |
| `LLM_API_KEY` | 是 | 只放 `.env`，不要提交 |
| `LLM_MODEL` | 是 | 模型名 |
| `LLM_REASONING_EFFORT` | 否 | `none` 可关掉推理模型的思考输出，省 token |
| `LLM_EXTRA_HEADERS` | 否 | JSON，例如某些网关要 `{"x-opencode-session":"..."}` |
| `AGENT_MAX_STEPS` | 否 | 循环上限，默认 8 |
| `ALLOWED_ROOT` | 否 | 文件工具沙箱根目录，默认当前目录 |

起 HTTP 服务：

```bash
uv run uvicorn agentlab.server:create_app --factory --reload --port 8000
curl -N -X POST localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"上海天气","session_id":"demo"}'
```

跑测试：

```bash
uv run pytest
uv run ruff check .
```

---

## 2. 目录结构（每一块对应一个知识点）

```
src/agentlab/
├── config.py         # 环境变量 -> Settings,密钥不进代码
├── llm.py            # ★ 直连 /chat/completions: messages / tools / SSE / usage
├── loop.py           # ★ Agent 主循环:while not done
├── tools/
│   ├── base.py       #   Tool = 名字 + 描述 + JSON Schema + 函数
│   ├── calc.py       #   AST 安全求值
│   ├── files.py      #   沙箱读文件 + 关键词搜索
│   └── weather.py    #   Open-Meteo
├── cli.py            # 流式 REPL
└── server.py         # FastAPI + SSE
scripts/              # 逐步教学脚本,建议按顺序跑一遍
├── step1_hello.py    # 最小的一次模型调用
├── step2_stream.py   # 手写 SSE 解析
├── step3_tools.py    # ★ 手工走一遍 tool calling 往返
├── step4_structured.py # structured output (json_schema)
└── step5_cache.py    # prompt caching 命中演示
tests/                # 全部离线,用 FakeLLM / MockTransport
docs/NOTES.md         # 五个概念的笔记
```

---

## 3. 两週计划（照着做）

### 第 1-2 天：把 API 摸熟

1. `uv run python scripts/step1_hello.py`
2. 打开 `llm.py`，对着 `_payload()` 看请求体：
   - `messages` 是 `[{role, content}]`，role 只有 `system/user/assistant/tool`
   - `tools` 是 JSON Schema 数组
3. 跑 `step2_stream.py`，观察 `data:` 一行行长什么样。
4. 跑 `step5_cache.py`，看第二次请求 `cached_tokens` 是否非 0。

**要能回答**：为什么 system prompt 放最前面？为什么工具结果要单独用 `role=tool`？

### 第 3-4 天：tool calling 往返

1. 跑 `step3_tools.py`，它会打印出 4 条消息的完整 JSON。
2. 对照 `loop.py` 的 `stream()`：哪一行把 `tool_calls` 存进历史？哪一行回填结果？
3. 故意改坏 `tools/calc.py` 的 schema（比如删掉 `required`），观察模型行为。

**要能回答**：如果模型返回的 `arguments` 不是合法 JSON，会发生什么？
（答案：`ToolArgumentError` → 错误文本回填 → 模型重试。见 `llm.py:parse_args`）

### 第 5-6 天：自己加工具

照着 `tools/weather.py` 加一个工具，建议二选一：

- `get_time(timezone)`：返回某时区当前时间
- `search_notes(keyword)`：在某个目录里 grep 文本

要求：写 schema、写函数、在 `default_registry()` 注册、加一条单测。
然后只改系统提示词，不加任何代码，看看 Agent 能不能自己学会用。

### 第 7-8 天：结构化输出 + 缓存

1. `step4_structured.py`：用 `response_format={"type":"json_schema"}` 让模型返回固定结构。
2. 把它接进 Agent：比如让模型最后输出 `{"answer": ..., "confidence": ...}`。
3. 读 `llm.py` 的 `Usage`：`cached_tokens / prompt_tokens` 就是命中率。
4. 实验：把 system prompt 从 200 字扩到 2000 字，连问三次，看命中率变化。

### 第 9-10 天：服务和测试

1. `uv run uvicorn agentlab.server:create_app --factory --port 8000`，用 curl 跑通 SSE。
2. 读 `tests/test_loop.py`：FakeLLM 如何"假装"模型返回工具调用。
3. 给 `/chat` 加一个 `max_steps` 参数，补测试。
4. 开源：`git init && git add . && git commit -m "feat: hand-written agent"`，
   然后 `gh repo create agent-lab --public --source=. --push`。

---

## 4. 你可能踩的坑

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `401` | Key 错 / base_url 少了 `/v1` | 检查 `.env` |
| 模型一直不调工具 | 工具描述太含糊 | 把 description 写具体，加"必须调用" |
| 工具参数 JSON 截断 | `max_tokens` 太小 | 调大，或让参数更短 |
| 流式没有 usage | 服务端不支持 `stream_options` | 代码会自动去掉重试 |
| 上下文越来越长 | 消息只增不减 | 这是第 2 阶段要学的记忆管理 |
| 读文件报"路径越界" | 沙箱限制 | 设 `ALLOWED_ROOT` 或传相对路径 |

---

## 5. 安全边界（现在就养成习惯）

- **密钥**：只从环境变量读，`.env` 已进 `.gitignore`。
- **文件**：`_resolve()` 先 `resolve()` 再 `is_relative_to()`，`../` 和软链接都逃不出去；
  `.env`、`.git`、私钥后缀直接拒绝。
- **计算**：AST 白名单，不用 `eval`，指数上限 1000。
- **循环**：`max_steps` 兜底，工具报错回填而不是抛给用户。
- **HTTP**：服务端按 session 存 Agent，生产环境要加鉴权和限流。

---

## 6. 已知限制 / 下一步

- 内存里的 session，重启即丢 → 换 SQLite / Redis
- 没有上下文压缩 → 第 2 阶段：token 预算 + 摘要
- 工具串行执行 → 可改 `asyncio.gather` 并行
- 没有重试策略分层、没有可观测性 → 第 3 阶段

---

## License

[MIT](LICENSE) © 2026 westson
