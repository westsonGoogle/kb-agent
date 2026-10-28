"""命令行入口:kb chat / ask / ingest / docs / reindex / serve。"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

from agentlab.config import Settings
from agentlab.llm import Usage

from .agent import build_agent
from .citations import CitationRegistry
from .config import SUPPORTED_SUFFIXES, KBSettings
from .kb import KnowledgeBase

USE_COLOR = sys.stdout.isatty() and not os.getenv("NO_COLOR")


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def _dim(text: str) -> str:
    return _c(text, "2")


def _cyan(text: str) -> str:
    return _c(text, "36")


def _yellow(text: str) -> str:
    return _c(text, "33")


def _red(text: str) -> str:
    return _c(text, "31")


def _green(text: str) -> str:
    return _c(text, "32")


def _usage_line(usage: Usage | None) -> str:
    if usage is None:
        return ""
    cached = f", 缓存 {usage.cached_tokens}" if usage.cached_tokens else ""
    return _dim(f"[tokens] 输入 {usage.prompt_tokens}{cached} / 输出 {usage.completion_tokens}")


def _print_sources(citations: CitationRegistry, answer: str = "") -> None:
    items = citations.items()
    if not items:
        return
    cited = {int(number) for number in re.findall(r"\[(\d+)\]", answer)}
    if cited:
        items = [item for item in items if item.cite in cited]
    print(_dim("\n参考来源"))
    for item in items[:8]:
        print(f"  {_yellow(f'[{item.cite}]')} {item.source} · {item.location}")
        print(_dim(f"      {item.snippet[:160]}"))


def _render_tool_call(name: str, arguments: dict | None) -> None:
    arguments = arguments or {}
    if name == "search_kb":
        print(f"\n{_cyan('  ↳ 检索')} {arguments.get('query', '')}", flush=True)
    elif name == "open_document":
        print(f"\n{_cyan('  ↳ 打开')} {arguments.get('source', '')}", flush=True)
    elif name == "list_documents":
        print(f"\n{_cyan('  ↳ 列出文档')}", flush=True)
    else:
        print(f"\n{_cyan('  ↳ 工具')} {name} {arguments}", flush=True)


def _render_tool_result(name: str, result: str) -> None:
    if name != "search_kb":
        return
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        return
    count = data.get("count", len(data.get("results", [])))
    print(_dim(f"    命中 {count} 段"), flush=True)


def _stream_turn(agent, citations: CitationRegistry, question: str) -> None:
    usage: Usage | None = None
    answer = ""
    citations.clear()
    print(_dim("助手 > "), end="", flush=True)
    try:
        for event in agent.stream(question):
            if event.type == "text":
                answer += event.text
                print(event.text, end="", flush=True)
            elif event.type == "tool_call":
                _render_tool_call(event.name, event.arguments)
            elif event.type == "tool_result":
                _render_tool_result(event.name, event.result)
            elif event.type == "usage":
                usage = event.usage
            elif event.type == "error":
                print(_red(f"\n[错误] {event.error}"))
    except KeyboardInterrupt:
        print(_red("\n[已中断]"))
        return
    print()
    _print_sources(citations, answer)
    line = _usage_line(usage)
    if line:
        print(line)


def cmd_chat(args: argparse.Namespace) -> int:
    kb = KnowledgeBase(KBSettings.from_env())
    if not kb.documents():
        print(_yellow("知识库是空的。先运行: kb ingest <文件或目录>"))
        return 0

    settings = Settings.from_env()
    citations = CitationRegistry()
    agent = build_agent(kb, citations, settings)

    print(f"{_cyan('kb-agent')} · {kb.chunk_count} 个片段 / {len(kb.documents())} 份文档")
    print(_dim("命令: /docs 文档  /reindex 重建索引  /reset 清空对话  /quit 退出\n"))

    while True:
        try:
            line = input(_green("你 > ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in {"/quit", "/exit", "/q"}:
            break
        if line == "/docs":
            _print_documents(kb)
            continue
        if line == "/reindex":
            stats = kb.reindex()
            print(_dim(f"已重建: {stats.indexed} 份文档 / {stats.chunks} 个片段"))
            for error in stats.errors:
                print(_red(f"  ! {error}"))
            continue
        if line == "/reset":
            agent.reset()
            citations.clear()
            print(_dim("对话已清空(索引不受影响)。"))
            continue
        if line == "/help":
            print(_dim("直接输入问题;工具会自动检索并给出来源。"))
            continue
        _stream_turn(agent, citations, line)
        print()

    print(_dim("再见。"))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    kb = KnowledgeBase(KBSettings.from_env())
    settings = Settings.from_env()
    citations = CitationRegistry()
    agent = build_agent(kb, citations, settings)
    _stream_turn(agent, citations, " ".join(args.question))
    print()
    return 0


def _print_documents(kb: KnowledgeBase) -> None:
    documents = kb.documents()
    if not documents:
        print(_dim("(空)"))
        return
    print(_dim(f"{'文档':<40} {'片段':>4}  {'大小':>8}"))
    for item in documents:
        print(f"{item.source:<40} {item.chunk_count:>4}  {item.size:>8}")
    print(_dim(f"共 {len(documents)} 份 / {kb.chunk_count} 个片段"))


def _copy_into_docs(source: Path, docs_dir: Path) -> Path:
    docs_dir.mkdir(parents=True, exist_ok=True)
    if source.resolve().is_relative_to(docs_dir):
        return source.resolve()

    destination = docs_dir / source.name
    if destination.exists() and destination.read_bytes() != source.read_bytes():
        for index in range(1, 1000):
            candidate = docs_dir / f"{source.stem}-{index}{source.suffix}"
            if not candidate.exists():
                destination = candidate
                break
    if not destination.exists():
        shutil.copy2(source, destination)
    return destination


def cmd_ingest(args: argparse.Namespace) -> int:
    source = Path(args.path).expanduser().resolve()
    if not source.exists():
        print(_red(f"路径不存在: {source}"))
        return 1

    kb = KnowledgeBase(KBSettings.from_env())
    settings = kb.settings

    if source.is_dir():
        files = [
            path
            for path in sorted(source.rglob("*"))
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_SUFFIXES
            and not any(part.startswith(".") for part in path.relative_to(source).parts)
        ]
    elif source.suffix.lower() in SUPPORTED_SUFFIXES:
        files = [source]
    else:
        print(_red(f"不支持的文件类型: {source.suffix}"))
        return 1

    if not files:
        print(_red("没有找到可导入的文件"))
        return 1

    failed = 0
    for path in files:
        try:
            target = _copy_into_docs(path, settings.docs_dir)
            info = kb.ingest(target)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(_red(f"! {path.name}: {exc}"))
            continue
        print(f"{_green('✓')} {info.source}  →  {info.chunk_count} 个片段")

    print(_dim(f"完成: {len(files) - failed} 成功 / {failed} 失败"))
    return 0 if failed == 0 else 1


def cmd_docs(args: argparse.Namespace) -> int:
    _print_documents(KnowledgeBase(KBSettings.from_env()))
    return 0


def cmd_reindex(args: argparse.Namespace) -> int:
    kb = KnowledgeBase(KBSettings.from_env())
    stats = kb.reindex()
    print(f"{_green('✓')} 重建完成: {stats.indexed} 份文档 / {stats.chunks} 个片段")
    if stats.removed:
        print(_dim(f"移除失效文档 {stats.removed} 份"))
    for error in stats.errors:
        print(_red(f"! {error}"))
    return 0 if not stats.errors else 1


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    print(_dim(f"http://{args.host}:{args.port}  (Ctrl+C 停止)"))
    uvicorn.run(
        "kbagent.server:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kb", description="带引用的本地知识库 Agent")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("chat", help="交互式问答(默认)")

    ask = subparsers.add_parser("ask", help="问一次就退出")
    ask.add_argument("question", nargs="+", help="问题")

    ingest = subparsers.add_parser("ingest", help="导入文件或目录到知识库")
    ingest.add_argument("path", help="文件或目录路径")

    subparsers.add_parser("docs", help="列出已索引文档")
    subparsers.add_parser("reindex", help="重建全部索引")

    serve = subparsers.add_parser("serve", help="启动 Web 界面")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "chat"
    handlers = {
        "chat": cmd_chat,
        "ask": cmd_ask,
        "ingest": cmd_ingest,
        "docs": cmd_docs,
        "reindex": cmd_reindex,
        "serve": cmd_serve,
    }
    return handlers[command](args)


if __name__ == "__main__":
    raise SystemExit(main())
