"""评测系统单元测试（离线：规则指标 + 用假LLM测裁判解析）"""
import os
import sys
import json
import pytest
from langchain_core.documents import Document

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_system import Evaluator


def make_doc(title, year="2020", content="some content"):
    return Document(page_content=content, metadata={"Title": title, "Year": year})


@pytest.fixture
def evaluator(tmp_path):
    """临时评测集文件"""
    dataset = [{
        "query": "测试问题",
        "type": "single-hop",
        "expected_papers": ["Attention Is All You Need"],
        "expected_keywords": ["注意力", "query"]
    }]
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    return Evaluator(dataset_path=str(path))


class TestRuleMetrics:
    def test_retrieval_exact(self, evaluator):
        papers = [make_doc("Attention Is All You Need")]
        assert evaluator.evaluate_retrieval(papers, ["Attention Is All You Need"]) == 1.0

    def test_retrieval_fuzzy(self, evaluator):
        """标题子串匹配（双向包含）"""
        papers = [make_doc("Attention Is All You Need (2017)")]
        assert evaluator.evaluate_retrieval(papers, ["Attention Is All You Need"]) == 1.0

    def test_retrieval_partial(self, evaluator):
        papers = [make_doc("Unrelated Paper")]
        assert evaluator.evaluate_retrieval(papers, ["Unrelated Paper", "BERT"]) == 0.5

    def test_retrieval_empty_expected(self, evaluator):
        assert evaluator.evaluate_retrieval([], []) == 1.0

    def test_answer_threshold(self, evaluator):
        """命中>=30%记1.0，否则返回原始比例"""
        kws = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        assert evaluator.evaluate_answer("包含了 a b c", kws) == 1.0   # 3/10 = 30%
        assert evaluator.evaluate_answer("包含了 a b", kws) == 0.2

    def test_answer_empty_expected(self, evaluator):
        assert evaluator.evaluate_answer("anything", []) == 1.0


class FakeLLM:
    """模拟裁判LLM，返回固定JSON"""
    def __init__(self, content):
        self.content = content

    def invoke(self, prompt):
        return type("Resp", (), {"content": self.content})()


class TestLLMJudge:
    def test_judge_parse_plain_json(self, evaluator):
        evaluator.llm = FakeLLM('{"answer_covered": 2, "expected_papers_covered": 1, "retrieved_relevant": 3}')
        item = evaluator.dataset[0]
        judge = evaluator.judge_item(item, "回答了注意力和query", [make_doc("X"), make_doc("Y"), make_doc("Z")])
        assert judge["answer_covered"] == 2
        assert judge["_kw_total"] == 2 and judge["_exp_total"] == 1

    def test_judge_parse_markdown_fence(self, evaluator):
        evaluator.llm = FakeLLM('```json\n{"answer_covered": 1, "expected_papers_covered": 0, "retrieved_relevant": 1}\n```')
        judge = evaluator.judge_item(evaluator.dataset[0], "r", [make_doc("X")])
        assert judge["retrieved_relevant"] == 1

    def test_judge_clamps_upper_bound(self, evaluator):
        """裁判输出超过上界时会被截断"""
        evaluator.llm = FakeLLM('{"answer_covered": 99, "expected_papers_covered": 99, "retrieved_relevant": 99}')
        judge = evaluator.judge_item(evaluator.dataset[0], "r", [make_doc("X")])
        assert judge["answer_covered"] == 2      # _kw_total=2
        assert judge["expected_papers_covered"] == 1
        assert judge["retrieved_relevant"] == 1  # 只有1篇论文

    def test_judge_failure_returns_none(self, evaluator):
        evaluator.llm = FakeLLM("这不是JSON")
        assert evaluator.judge_item(evaluator.dataset[0], "r", []) is None

    def test_no_llm_returns_none(self, evaluator):
        evaluator.llm = None
        assert evaluator.judge_item(evaluator.dataset[0], "r", []) is None


class TestCompareAndExport:
    def test_export_backup_and_compare(self, evaluator, tmp_path, capsys):
        out = str(tmp_path / "results.json")
        results = [{"query": "q", "type": "single-hop", "retrieval_acc": 0.5,
                    "answer_acc": 1.0, "judge": None, "steps": 1, "paper_count": 4, "elapsed": 1.0}]
        evaluator.export_results(results, out)
        assert os.path.exists(out)

        # 第二次导出应生成备份，且对比功能正常
        results2 = [dict(results[0], retrieval_acc=0.8)]
        evaluator.export_results(results2, out)
        assert os.path.exists(out.replace(".json", "_prev.json"))
        evaluator.compare_with_file(results2, out.replace(".json", "_prev.json"))
        captured = capsys.readouterr().out
        assert "50.0%" in captured and "80.0%" in captured
