# 五个概念,一次讲透

这份笔记对应第 0 阶段的五个知识点。建议对照代码读,读完做文末的自测题。

---

## 1. messages:对话就是一个数组

```json
[
  {"role": "system",    "content": "你是助手"},
  {"role": "user",      "content": "北京天气?"},
  {"role": "assistant", "content": "", "tool_calls": [...]},
  {"role": "tool",      "tool_call_id": "call_1", "content": "..."},
  {"role": "assistant", "content": "北京现在 18°C"}
]
```

只有四种 role：

| role | 谁写的 | 说明 |
| --- | --- | --- |
| `system` | 你 | 规则、人设、工具使用约定。放最前面 |
| `user` | 用户 | 输入 |
| `assistant` | 模型 | 回答；要工具时带 `tool_calls` |
| `tool` | 你的代码 | 工具结果，必须带 `tool_call_id` |

三条硬规则：

1. **模型是无状态的**。"记住上下文"= 每次把完整 messages 再发一遍。
2. `role=tool` 的消息必须紧跟在带 `tool_calls` 的 assistant 消息后面，且 `tool_call_id` 一一对应。
   顺序错了，OpenAI 会直接 400。
3. 上下文的成本是 O(轮数²)：第 N 轮要重发前 N-1 轮。这就是 prompt caching 存在的理由。

> 代码：`loop.py:stream()` 里 `self.messages.append(...)` 的三处。

---

## 2. tool calling:模型只负责"说",你负责"做"

完整往返（见 `scripts/step3_tools.py`）：

```
你  ── tools=[schema...] + messages ──▶ 模型
模型 ── finish_reason="tool_calls" + tool_calls[{id,name,arguments}] ──▶ 你
你  ── 执行函数，append {role:"tool", tool_call_id, content} ──▶ 模型
模型 ── 最终自然语言回答 ──▶ 你
```

关键认知：

- 模型**不会执行任何东西**，它只是输出"我想调用 get_weather，参数是 {...}"。
- `arguments` 是**字符串**，不是对象。可能非法、可能缺字段、可能多字段 → 必须容错并回填错误。
- assistant 那条消息要**原样**存回历史（包括 `tool_calls`），否则下一轮模型不知道自己在等什么。
- 一次响应可以包含多个 tool_calls（并行工具调用），按顺序执行、逐条回填即可。
- schema 里的 `description` 就是 prompt。工具选错，先改描述，再改代码。

失败模式：

| 模型行为 | 你的处理 |
| --- | --- |
| 调不存在的工具 | 回填 `错误: 没有这个工具...`，让它重选 |
| 参数不是 JSON | 回填解析错误，让它重发 |
| 缺必填参数 | 回填缺哪个 |
| 工具内部异常 | 捕获成文本回填，别让进程挂掉 |
| 一直循环调工具 | `max_steps` 强制刹车 |

> 代码：`tools/base.py:execute()` 永不抛异常；`loop.py` 的 `for call in response.tool_calls`。

---

## 3. streaming:把 JSON 切成小块发给你

响应体是 SSE 文本，不是 JSON：

```
data: {"choices":[{"delta":{"role":"assistant","content":"你"}}]}

data: {"choices":[{"delta":{"content":"好"}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"ci"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\":\"北京\"}"}}]},"finish_reason":"tool_calls"}]}

data: {"choices":[],"usage":{...}}

data: [DONE]
```

要点：

- 只看 `choices[0].delta`，把 `content` 片段拼起来。
- **工具参数也会被切碎**，按 `index` 分组后用 `+=` 拼接，流结束才是完整 JSON。
- `usage` 默认不返回，要加 `stream_options: {"include_usage": true}`；有的兼容服务不认，要能降级重试。
- 好处是首 token 延迟（TTFT）低；代价是解析、错误重试、断线续传都变复杂。
- 推理型模型还会有 `reasoning_content`，和正式回答分开累积。

> 代码：`llm.py:_parse_sse()`，以及 `scripts/step2_stream.py` 的手写版。

---

## 4. structured output:让模型变成函数

三种做法，可靠性从高到低：

1. **`response_format={"type":"json_schema", ...}`**
   服务端约束解码，schema 里要 `additionalProperties: false`，且所有字段都写进 `required`
   （OpenAI strict 模式的要求）。
2. **`response_format={"type":"json_object"}`**
   只保证是合法 JSON，不保证字段。仍需自己校验。
3. **Prompt 里贴 schema + 手动 parse**
   兼容性最好，可靠性最差。必须处理 ```json 包裹、前后废话、字段缺失。

工程上永远要做**校验 + 修复**：`json.loads` → 检查必填字段/类型 → 不合格就让模型重试一次。
不要假设模型一定听话。

> 代码：`scripts/step4_structured.py`（含 400 降级路径）。

---

## 5. prompt caching:省钱的唯一正确姿势

原理：服务端对**输入前缀**做 KV cache。下一次请求如果前缀逐字节相同，就直接复用，
按缓存价计费（通常便宜 5-10 倍），首 token 也更快。

```
[system 2000 字][工具定义][历史 20 轮][本轮问题]
└────────── 稳定前缀,可缓存 ──────────┘ └ 变化 ┘
```

规则：

- **前缀必须完全一致**：一个空格、一个时间戳、工具顺序变化都会导致 miss。
- 稳定内容放前面（system、工具、长文档），变化内容放后面（用户问题）。
- 命中量看 `usage.prompt_tokens_details.cached_tokens`（DeepSeek 是 `prompt_cache_hit_tokens`）。
- OpenAI 大约 1024 token 起、按 128 对齐；DeepSeek 64 token 起；Anthropic 要显式
  `cache_control: {"type":"ephemeral"}` 打断点。
- 缓存有 TTL（几分钟到几小时），冷启动第一次必然 miss。

实测（`scripts/step5_cache.py`）：

```
第 1 次: 输入 2983, 命中 0      (0.0%)
第 2 次: 输入 2983, 命中 2944   (98.7%)
第 3 次: 输入 2980, 命中 2944   (98.8%)
```

Agent 场景里，工具 schema 和 system prompt 每次请求都一样，**天生适合缓存**。
所以别在 system 里塞 `当前时间: 2026-10-28 09:45:03` 这种东西。

---

## 6. 最后：循环的不变量

```python
while not done:
    response = model(messages, tools)
    messages.append(assistant_message(response))
    if not response.tool_calls:
        done = True
    else:
        for call in response.tool_calls:
            messages.append(tool_message(call, execute(call)))
```

不管你以后用 LangGraph、OpenAI Agents SDK 还是自己写，**骨架永远是这 10 行**。
框架帮你加的是：持久化、并行、重试、可观测性、多 Agent 编排、人类审批。

---

## 自测题

1. 为什么 `role=tool` 的消息必须带 `tool_call_id`？不带会怎样？
2. 流式返回的 `tool_calls[0].function.arguments` 为什么不能每个 chunk 都 `json.loads`？
3. 想让 prompt cache 命中率最大化，system prompt 里应该避免放什么？
4. 模型连续 8 次调用同一个工具且参数相同，你怎么在循环层面止损？
5. `json_schema` 的 strict 模式为什么要求 `additionalProperties: false`？
6. 如果用户让你读 `../../.ssh/id_rsa`，你的工具链会在哪一层拦住？

答案都能在前面的代码里找到。
