# uv 与 FastAPI 速查

## 为什么用 uv 而不是 pip + venv

传统流程要手动做三件事：`python -m venv .venv`、`source .venv/bin/activate`、`pip install -r requirements.txt`。
uv 把这三步合成一条命令，并且自带锁文件：

- `uv init` 创建 `pyproject.toml`
- `uv add httpx` 添加依赖并写入 `pyproject.toml` + `uv.lock`
- `uv sync` 严格按锁文件安装，自动创建 `.venv`
- `uv run pytest` 在项目环境里执行命令，不需要 activate
- `uv lock --upgrade` 升级锁文件

关键区别：`uv.lock` 记录的是完整解析结果（含传递依赖和哈希），所以 CI 和同事机器上装出来的版本完全一致。
pip 的 `requirements.txt` 默认只锁直接依赖，容易出现“我这里是好的”。

生产建议：

1. 提交 `uv.lock`，不要提交 `.venv`。
2. CI 用 `uv sync --locked`，锁文件过期直接失败。
3. 不要在 Dockerfile 里用 `pip install -e .` 绕过锁文件。

## FastAPI 的流式接口

普通接口返回值会被框架序列化成 JSON，等全部算完才返回。流式接口要用 `StreamingResponse`：

```python
from fastapi.responses import StreamingResponse


def event_stream():
    for token in generate():
        yield f"data: {json.dumps({'type': 'text', 'text': token})}\n\n"


return StreamingResponse(
    event_stream(),
    media_type="text/event-stream",
    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
)
```

三个容易踩的坑：

1. 响应必须是 `text/event-stream`，每条消息以两个换行结束。
2. 反向代理（Nginx）会缓冲响应，需要 `X-Accel-Buffering: no` 或关闭 `proxy_buffering`。
3. 生成器里抛异常时连接已经建立，不能再改状态码，只能发一条 `{"type":"error"}` 事件。

## 同步 vs 异步

FastAPI 的 `def` 接口跑在线程池，`async def` 跑在事件循环。同步的 Agent 循环（httpx 同步调用）放在 `def` 里最安全；
如果写成 `async def` 又调用阻塞函数，会把整个事件循环卡住，所有请求一起变慢。

## SSE 与 WebSocket 怎么选

- 单向推送（模型输出、进度条）→ SSE，协议简单，自带断线重连。
- 双向实时（协同编辑、语音）→ WebSocket。
- 本项目的聊天是“客户端 POST 一次、服务端流式返回”，SSE 足够，而且能用普通 HTTP 调试。
