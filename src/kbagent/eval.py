"""评测:30 条问题跑一遍,量化召回、引用、忠实度和拒答。

设计原则:
- 能自动算的绝不靠感觉:来源召回、引用召回直接由字符串匹配得出。
- 语义质量交给 LLM 判分,但给出明确 rubric 和参考要点,降低随机性。
- 所有结果落盘为 JSON + Markdown,方便对比不同版本。
"""

from __future__ import annotations

import json
import re
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentlab.config import Settings
from agentlab.llm import LLMClient, LLMError, Usage

from .agent import build_agent
from .citations import CitationRegistry
from .config import KBSettings
from .kb import KnowledgeBase

JUDGE_SYSTEM = """你是严格的 RAG 评测员。根据「问题」「参考要点」和「候选回答及引用」打分。

只输出 JSON:
{"score": 0, "faithful": true, "refused": false, "reason": "一句话理由"}

评分标准:
- 2: 覆盖所有参考要点,没有编造,引用能支撑结论。
- 1: 覆盖部分要点,或有个别无依据的补充。
- 0: 答错、答非所问,或大量编造。

判分只看语义是否覆盖、是否有依据,不要因为措辞、句式或详略不同而扣分。
引用编号不需要逐字对应,但必须能支撑相邻结论。

faithful=true 表示回答中的事实都能在引用片段中找到依据。
若题目标记为「资料中本来没有答案」,此时:
- 2: 明确说明资料没有,且不编造具体信息。
- 1: 含糊其辞或部分猜测。
- 0: 编造了具体信息。
refused=true 表示回答明确拒绝/声明资料不足。"""


@dataclass(slots=True)
class EvalQuestion:
    id: str
    question: str
    category: str = "general"
    expected_sources: list[str] = field(default_factory=list)
    key_points: list[str] = field(default_factory=list)
    must_refuse: bool = False


@dataclass(slots=True)
class JudgeResult:
    score: int
    faithful: bool
    refused: bool
    reason: str = ""


@dataclass(slots=True)
class EvalResult:
    question: EvalQuestion
    answer: str
    retrieved_sources: list[str]
    cited_sources: list[str]
    cited_details: list[dict[str, Any]]
    retrieval_recall: float
    citation_recall: float
    usage: Usage
    elapsed: float
    judge: JudgeResult | None = None
    error: str = ""


def load_questions(path: Path) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {lineno} 行不是合法 JSON: {exc}") from exc
        try:
            question = EvalQuestion(
                id=str(data["id"]),
                question=str(data["question"]),
                category=str(data.get("category", "general")),
                expected_sources=[str(item) for item in data.get("expected_sources", [])],
                key_points=[str(item) for item in data.get("key_points", [])],
                must_refuse=bool(data.get("must_refuse", False)),
            )
        except KeyError as exc:
            raise ValueError(f"第 {lineno} 行缺少字段 {exc}") from exc

        if not question.id or not question.question:
            raise ValueError(f"第 {lineno} 行 id/question 不能为空")
        if question.must_refuse and question.expected_sources:
            raise ValueError(f"第 {lineno} 行:must_refuse 的题不应有 expected_sources")
        questions.append(question)

    if not questions:
        raise ValueError("评测集为空")
    return questions


def parse_citations(text: str) -> list[int]:
    return sorted({int(number) for number in re.findall(r"\[(\d+)\]", text)})


def _coverage(expected: set[str], actual: set[str]) -> float:
    if not expected:
        return 1.0
    return len(expected & actual) / len(expected)


def run_agent(
    kb: KnowledgeBase,
    settings: Settings,
    question: EvalQuestion,
    *,
    llm: Any | None = None,
    retries: int = 2,
    retry_delay: float = 1.0,
) -> EvalResult:
    """跑一道题。网关 401/500 这类瞬时错误重试,避免把基础设施故障算进模型分数。"""
    last: EvalResult | None = None
    for attempt in range(retries + 1):
        result = _run_agent_once(kb, settings, question, llm=llm)
        if not result.error:
            return result
        last = result
        if attempt < retries:
            time.sleep(retry_delay * (attempt + 1))
    assert last is not None
    return last


def _run_agent_once(
    kb: KnowledgeBase,
    settings: Settings,
    question: EvalQuestion,
    *,
    llm: Any | None = None,
) -> EvalResult:
    citations = CitationRegistry()
    agent = build_agent(kb, citations, settings)
    if llm is not None:
        agent.llm = llm

    parts: list[str] = []
    usage = Usage()
    error = ""
    started = time.perf_counter()
    try:
        for event in agent.stream(question.question):
            if event.type == "text":
                parts.append(event.text)
            elif event.type == "usage" and event.usage is not None:
                usage.prompt_tokens += event.usage.prompt_tokens
                usage.completion_tokens += event.usage.completion_tokens
                usage.total_tokens += event.usage.total_tokens
                usage.cached_tokens += event.usage.cached_tokens
            elif event.type == "error":
                error = event.error
    except Exception as exc:  # noqa: BLE001 - 评测要记录失败而不是中断整批
        error = str(exc)
    elapsed = time.perf_counter() - started

    answer = "".join(parts).strip()
    items = {item.cite: item for item in citations.items()}
    cited_numbers = parse_citations(answer)
    cited_sources = [items[number].source for number in cited_numbers if number in items]
    cited_details = [items[number].to_dict() for number in cited_numbers if number in items]
    retrieved_sources = [item.source for item in citations.items()]

    expected = set(question.expected_sources)
    return EvalResult(
        question=question,
        answer=answer,
        retrieved_sources=retrieved_sources,
        cited_sources=cited_sources,
        cited_details=cited_details,
        retrieval_recall=_coverage(expected, set(retrieved_sources)),
        citation_recall=_coverage(expected, set(cited_sources)),
        usage=usage,
        elapsed=elapsed,
        error=error,
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_judge_json(text: str) -> JudgeResult:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, re.S)
    if not match:
        raise ValueError(f"判分结果不是 JSON: {text[:200]}")
    data = json.loads(match.group(0))
    score = max(0, min(2, int(data.get("score", 0))))
    return JudgeResult(
        score=score,
        faithful=_as_bool(data.get("faithful", False)),
        refused=_as_bool(data.get("refused", False)),
        reason=str(data.get("reason", "")),
    )


def judge_answer(
    llm: LLMClient,
    question: EvalQuestion,
    answer: str,
    cited_details: list[dict[str, Any]],
) -> JudgeResult:
    evidence = (
        "\n\n".join(
            f"[{item['cite']}] {item['source']} {item['location']}\n{item['snippet']}"
            for item in cited_details[:6]
        )
        or "(没有引用任何来源)"
    )

    if question.must_refuse:
        task = "注意:资料库中本来就没有这道题的答案,正确行为是明确说明资料没有。"
    else:
        task = "参考要点:\n" + "\n".join(f"- {point}" for point in question.key_points)

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"问题:{question.question}\n\n{task}\n\n"
                f"候选回答:\n{answer[:1500]}\n\n引用片段:\n{evidence[:3000]}"
            ),
        },
    ]
    try:
        response = llm.chat(
            messages,
            temperature=0,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
    except LLMError:
        response = llm.chat(messages, temperature=0, max_tokens=500)

    try:
        return parse_judge_json(response.content)
    except (ValueError, json.JSONDecodeError) as exc:
        return JudgeResult(score=0, faithful=False, refused=False, reason=f"判分解析失败: {exc}")


def evaluate(
    kb: KnowledgeBase,
    settings: Settings,
    questions: list[EvalQuestion],
    *,
    limit: int | None = None,
    use_judge: bool = True,
    judge_fn: Callable[[EvalQuestion, str, list[dict[str, Any]]], JudgeResult] | None = None,
    llm_factory: Callable[[], Any] | None = None,
    progress: Callable[[int, int, EvalResult], None] | None = None,
    retries: int = 2,
) -> list[EvalResult]:
    selected = questions[:limit] if limit else questions
    judge_llm = LLMClient(settings) if use_judge and judge_fn is None else None

    results: list[EvalResult] = []
    for index, question in enumerate(selected, start=1):
        result = run_agent(
            kb,
            settings,
            question,
            llm=llm_factory() if llm_factory is not None else None,
            retries=retries,
        )
        if use_judge:
            if judge_fn is not None:
                result.judge = judge_fn(question, result.answer, result.cited_details)
            elif judge_llm is not None:
                result.judge = judge_answer(
                    judge_llm, question, result.answer, result.cited_details
                )
        results.append(result)
        if progress is not None:
            progress(index, len(selected), result)
    return results


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _optional_mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _fmt_pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _fmt_score(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def summarize(results: list[EvalResult]) -> dict[str, Any]:
    valid = [item for item in results if not item.error]
    answerable = [item for item in valid if not item.question.must_refuse]
    unanswerable = [item for item in valid if item.question.must_refuse]
    judged = [item for item in results if item.judge is not None]
    judged_answerable = [item for item in answerable if item.judge is not None]
    judged_unanswerable = [item for item in unanswerable if item.judge is not None]

    scores = [float(item.judge.score) for item in judged_answerable if item.judge]
    answer_score = _optional_mean(scores)
    errors = sum(1 for item in results if item.error)
    return {
        "questions": len(results),
        "valid": len(valid),
        "errors": errors,
        "error_rate": errors / len(results) if results else 0.0,
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "retrieval_recall": _optional_mean([item.retrieval_recall for item in answerable]),
        "citation_recall": _optional_mean([item.citation_recall for item in answerable]),
        "citation_rate": _optional_mean(
            [1.0 if item.cited_sources else 0.0 for item in answerable]
        ),
        "answer_score": answer_score,
        "answer_score_pct": answer_score / 2 * 100 if answer_score is not None else None,
        "faithfulness": _optional_mean(
            [1.0 if item.judge and item.judge.faithful else 0.0 for item in judged_answerable]
        ),
        "refusal_accuracy": _optional_mean(
            [1.0 if item.judge and item.judge.refused else 0.0 for item in judged_unanswerable]
        ),
        "avg_latency": _mean([item.elapsed for item in results]),
        "prompt_tokens": sum(item.usage.prompt_tokens for item in results),
        "completion_tokens": sum(item.usage.completion_tokens for item in results),
        "cached_tokens": sum(item.usage.cached_tokens for item in results),
        "judged": len(judged),
    }


def summarize_by_category(results: list[EvalResult]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[EvalResult]] = defaultdict(list)
    for item in results:
        groups[item.question.category].append(item)
    return {category: summarize(items) for category, items in sorted(groups.items())}


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def render_markdown(
    results: list[EvalResult],
    summary: dict[str, Any],
    meta: dict[str, Any],
) -> str:
    questions_line = (
        f"| 题目数 | {summary['questions']}"
        f"(有效 {summary['valid']} / 可回答 {summary['answerable']} / "
        f"不可回答 {summary['unanswerable']}) |"
    )
    score_pct = summary["answer_score_pct"]
    score_pct_text = "—" if score_pct is None else f"{score_pct:.0f}%"
    score_line = f"| 回答质量(0-2) | {_fmt_score(summary['answer_score'])} / 2({score_pct_text}) |"
    token_line = (
        f"| Token | 输入 {summary['prompt_tokens']}"
        f"(缓存 {summary['cached_tokens']}) / 输出 {summary['completion_tokens']} |"
    )

    lines = [
        "# kb-agent 评测报告",
        "",
        f"- 时间:{meta.get('timestamp', '')}",
        f"- 模型:{meta.get('model', '')}",
        f"- 数据集:{meta.get('dataset', '')}",
        f"- 知识库:{meta.get('documents', 0)} 份文档 / {meta.get('chunks', 0)} 个片段",
        "",
        "## 总览",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        questions_line,
        f"| 来源召回率 | {_fmt_pct(summary['retrieval_recall'])} |",
        f"| 引用召回率 | {_fmt_pct(summary['citation_recall'])} |",
        f"| 有引用回答占比 | {_fmt_pct(summary['citation_rate'])} |",
        score_line,
        f"| 忠实度 | {_fmt_pct(summary['faithfulness'])} |",
        f"| 拒答准确率 | {_fmt_pct(summary['refusal_accuracy'])} |",
        f"| 平均耗时 | {summary['avg_latency']:.2f}s |",
        token_line,
        f"| 错误 | {summary['errors']}({summary['error_rate']:.1%}) |",
        "",
        "## 分类",
        "",
        "| 分类 | 题数 | 召回 | 引用召回 | 回答分 | 忠实度 | 拒答 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for category, stats in summarize_by_category(results).items():
        lines.append(
            f"| {category} | {stats['questions']} | "
            f"{_fmt_pct(stats['retrieval_recall'])} | "
            f"{_fmt_pct(stats['citation_recall'])} | "
            f"{_fmt_score(stats['answer_score'])} | "
            f"{_fmt_pct(stats['faithfulness'])} | "
            f"{_fmt_pct(stats['refusal_accuracy'])} |"
        )

    lines.extend(
        [
            "",
            "## 明细",
            "",
            "| # | 题目 | 分类 | 来源召回 | 引用召回 | 分数 | 忠实 | 拒答 | 耗时 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for item in results:
        judge = item.judge
        recall = "—" if item.question.must_refuse else f"{item.retrieval_recall:.0%}"
        cite_recall = "—" if item.question.must_refuse else f"{item.citation_recall:.0%}"
        score = str(judge.score) if judge else "—"
        faithful = "✓" if judge and judge.faithful else ("✗" if judge else "—")
        refused = "✓" if judge and judge.refused else ("✗" if judge else "—")
        question = _cell(item.question.question)[:46]
        lines.append(
            f"| {item.question.id} | {question} | {item.question.category} | {recall} | "
            f"{cite_recall} | {score} | {faithful} | {refused} | {item.elapsed:.1f}s |"
        )

    failures = [item for item in results if item.judge is not None and item.judge.score < 2]
    if failures:
        lines.extend(["", "## 未满分样例", ""])
        for item in failures[:10]:
            judge = item.judge
            assert judge is not None
            lines.extend(
                [
                    f"### {item.question.id} · {item.question.question}",
                    "",
                    f"- 判分:{judge.score}/2,忠实:{judge.faithful},拒答:{judge.refused}",
                    f"- 理由:{judge.reason}",
                    f"- 回答节选:{_cell(item.answer)[:300]}",
                    "",
                ]
            )

    if summary["errors"]:
        lines.extend(["", "## 错误", ""])
        for item in results:
            if item.error:
                lines.append(f"- `{item.question.id}` {item.error}")

    return "\n".join(lines) + "\n"


def default_meta(
    settings: Settings,
    kb: KnowledgeBase,
    dataset: Path,
    timestamp: str,
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "model": settings.model,
        "dataset": str(dataset),
        "documents": len(kb.documents()),
        "chunks": kb.chunk_count,
    }


def write_reports(
    results: list[EvalResult],
    *,
    markdown_path: Path,
    meta: dict[str, Any],
) -> tuple[Path, Path]:
    summary = summarize(results)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(results, summary, meta), encoding="utf-8")

    json_path = markdown_path.with_suffix(".json")
    payload = {
        "meta": meta,
        "summary": summary,
        "categories": summarize_by_category(results),
        "results": [
            {
                "id": item.question.id,
                "question": item.question.question,
                "category": item.question.category,
                "must_refuse": item.question.must_refuse,
                "answer": item.answer,
                "retrieved_sources": item.retrieved_sources,
                "cited_sources": item.cited_sources,
                "retrieval_recall": item.retrieval_recall,
                "citation_recall": item.citation_recall,
                "elapsed": item.elapsed,
                "error": item.error,
                "judge": (
                    {
                        "score": item.judge.score,
                        "faithful": item.judge.faithful,
                        "refused": item.judge.refused,
                        "reason": item.judge.reason,
                    }
                    if item.judge
                    else None
                ),
            }
            for item in results
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return markdown_path, json_path


# 供 CLI 默认使用
def kb_from_settings(settings: KBSettings | None = None) -> KnowledgeBase:
    return KnowledgeBase(settings or KBSettings.from_env())
