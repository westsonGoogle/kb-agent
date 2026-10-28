from __future__ import annotations

import json
from pathlib import Path

import pytest
from agentlab.config import Settings
from agentlab.llm import ToolCall
from fakes import FakeLLM, text_script, tool_script

from kbagent.config import KBSettings
from kbagent.eval import (
    EvalQuestion,
    JudgeResult,
    evaluate,
    load_questions,
    parse_citations,
    parse_judge_json,
    render_markdown,
    run_agent,
    summarize,
    write_reports,
)
from kbagent.kb import KnowledgeBase

SETTINGS = Settings(api_key="test", base_url="http://llm.test/v1", model="fake-model")


@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    settings = KBSettings(
        data_dir=tmp_path,
        docs_dir=tmp_path / "docs",
        db_path=tmp_path / "kb.db",
        chunk_chars=400,
        chunk_overlap=60,
    )
    knowledge = KnowledgeBase(settings)
    (settings.docs_dir / "caching.md").write_text(
        "prompt cache 要求前缀完全一致。\n时间戳会破坏缓存命中。", encoding="utf-8"
    )
    knowledge.reindex()
    return knowledge


def make_question(**overrides) -> EvalQuestion:
    data = {
        "id": "q01",
        "question": "缓存怎么命中?",
        "category": "agent",
        "expected_sources": ["caching.md"],
        "key_points": ["前缀一致"],
    }
    data.update(overrides)
    return EvalQuestion(**data)


def make_answer_scripts() -> list:
    return [
        tool_script(ToolCall(id="call_1", name="search_kb", arguments='{"query": "缓存 命中"}')),
        text_script("前缀必须完全一致 [1]。"),
    ]


# ---------------------------------------------------------------------------
# 数据集与解析
# ---------------------------------------------------------------------------


def test_load_questions_skips_comments(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text(
        '# comment\n{"id": "a", "question": "问题 A"}\n\n'
        '{"id": "b", "question": "问题 B", "must_refuse": true}\n',
        encoding="utf-8",
    )
    questions = load_questions(path)
    assert [question.id for question in questions] == ["a", "b"]
    assert questions[1].must_refuse is True


def test_load_questions_rejects_refuse_with_sources(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text(
        '{"id": "a", "question": "q", "must_refuse": true, "expected_sources": ["x.md"]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must_refuse"):
        load_questions(path)


def test_load_questions_rejects_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="不是合法 JSON"):
        load_questions(path)


def test_bundled_dataset_shape() -> None:
    questions = load_questions(Path("evals/questions.jsonl"))
    assert len(questions) == 30
    assert len({question.id for question in questions}) == 30
    assert sum(question.must_refuse for question in questions) == 3
    assert all(question.key_points for question in questions if not question.must_refuse)


def test_parse_citations() -> None:
    assert parse_citations("结论 [1],补充 [3][1],无效 [99]") == [1, 3, 99]
    assert parse_citations("没有引用") == []


def test_parse_judge_json_variants() -> None:
    direct = parse_judge_json('{"score": 2, "faithful": true, "refused": false, "reason": "ok"}')
    assert direct == JudgeResult(2, True, False, "ok")

    fenced = parse_judge_json('```json\n{"score": 9, "faithful": "true", "refused": "false"}\n```')
    assert fenced.score == 2
    assert fenced.faithful is True
    assert fenced.refused is False

    with pytest.raises(ValueError):
        parse_judge_json("不是 JSON")


# ---------------------------------------------------------------------------
# 跑单题与整批
# ---------------------------------------------------------------------------


def test_run_agent_collects_recall_and_citations(kb: KnowledgeBase) -> None:
    question = make_question()
    result = run_agent(kb, SETTINGS, question, llm=FakeLLM(make_answer_scripts()))

    assert result.error == ""
    assert "前缀必须完全一致" in result.answer
    assert "caching.md" in result.retrieved_sources
    assert result.cited_sources == ["caching.md"]
    assert result.retrieval_recall == 1.0
    assert result.citation_recall == 1.0
    assert result.usage.prompt_tokens > 0


def test_run_agent_missing_source_lowers_recall(kb: KnowledgeBase) -> None:
    question = make_question(expected_sources=["caching.md", "missing.md"])
    result = run_agent(kb, SETTINGS, question, llm=FakeLLM(make_answer_scripts()))
    assert result.retrieval_recall == 0.5
    assert result.citation_recall == 0.5


def test_evaluate_and_summarize(kb: KnowledgeBase) -> None:
    question = make_question()
    progress: list[tuple[int, int]] = []

    results = evaluate(
        kb,
        SETTINGS,
        [question],
        judge_fn=lambda q, answer, cites: JudgeResult(2, True, False, "覆盖完整"),
        llm_factory=lambda: FakeLLM(make_answer_scripts()),
        progress=lambda index, total, result: progress.append((index, total)),
    )

    assert len(results) == 1
    assert results[0].judge is not None
    assert results[0].judge.score == 2
    assert progress == [(1, 1)]

    summary = summarize(results)
    assert summary["retrieval_recall"] == 1.0
    assert summary["citation_recall"] == 1.0
    assert summary["answer_score"] == 2.0
    assert summary["faithfulness"] == 1.0
    assert summary["refusal_accuracy"] is None
    assert summary["errors"] == 0


def test_unanswerable_refusal_metric(kb: KnowledgeBase) -> None:
    question = make_question(
        id="q28",
        question="公司年假几天?",
        category="unanswerable",
        expected_sources=[],
        key_points=[],
        must_refuse=True,
    )
    results = evaluate(
        kb,
        SETTINGS,
        [question],
        judge_fn=lambda q, answer, cites: JudgeResult(2, True, True, "明确拒答"),
        llm_factory=lambda: FakeLLM([text_script("资料里没有提到。")]),
    )
    summary = summarize(results)
    assert summary["refusal_accuracy"] == 1.0
    assert summary["answerable"] == 0


def test_evaluate_limit(kb: KnowledgeBase) -> None:
    questions = [make_question(id="a"), make_question(id="b")]
    results = evaluate(
        kb,
        SETTINGS,
        questions,
        limit=1,
        use_judge=False,
        llm_factory=lambda: FakeLLM(make_answer_scripts()),
    )
    assert len(results) == 1
    assert results[0].question.id == "a"


def test_run_agent_retries_transient_error(kb: KnowledgeBase) -> None:
    class FlakyLLM:
        def __init__(self) -> None:
            self.calls = 0

        def stream_chat(self, messages, tools=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("upstream 500")
            yield from text_script("重试后成功 [1]")

    llm = FlakyLLM()
    result = run_agent(kb, SETTINGS, make_question(), llm=llm, retries=1, retry_delay=0)

    assert llm.calls == 2
    assert result.error == ""
    assert "重试后成功" in result.answer


def test_summary_excludes_errored_questions(kb: KnowledgeBase) -> None:
    result = run_agent(
        kb,
        SETTINGS,
        make_question(),
        llm=FakeLLM(make_answer_scripts()),
    )
    result.error = "upstream 500"

    summary = summarize([result])
    assert summary["valid"] == 0
    assert summary["errors"] == 1
    assert summary["retrieval_recall"] is None
    assert summary["answer_score"] is None


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def test_report_contains_metrics_and_failure(kb: KnowledgeBase) -> None:
    question = make_question()
    results = evaluate(
        kb,
        SETTINGS,
        [question],
        judge_fn=lambda q, answer, cites: JudgeResult(1, False, False, "漏了一个要点"),
        llm_factory=lambda: FakeLLM(make_answer_scripts()),
    )
    summary = summarize(results)
    markdown = render_markdown(
        results,
        summary,
        {"timestamp": "2026-10-28 12:00", "model": "fake", "dataset": "x.jsonl"},
    )
    assert "来源召回率" in markdown
    assert "| 拒答准确率 | — |" in markdown
    assert "未满分样例" in markdown
    assert "漏了一个要点" in markdown


def test_write_reports(kb: KnowledgeBase, tmp_path: Path) -> None:
    question = make_question()
    results = evaluate(
        kb,
        SETTINGS,
        [question],
        judge_fn=lambda q, answer, cites: JudgeResult(2, True, False, "ok"),
        llm_factory=lambda: FakeLLM(make_answer_scripts()),
    )
    markdown_path, json_path = write_reports(
        results,
        markdown_path=tmp_path / "reports" / "eval.md",
        meta={"model": "fake"},
    )

    assert markdown_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["questions"] == 1
    assert payload["results"][0]["id"] == "q01"
    assert payload["results"][0]["judge"]["score"] == 2
