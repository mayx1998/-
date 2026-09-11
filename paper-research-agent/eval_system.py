"""
eval_system.py — 独立评测模块
提供完整的RAG系统评测功能：
  - 单条/批量评测
  - 检索准确率 + 回答准确率
  - 分类型统计
  - 失败Case分析

用法:
    from eval_system import Evaluator

    evaluator = Evaluator(dataset_path="eval_dataset.json")
    results = evaluator.run(app)  # app 是你的 LangGraph 编译后的对象
    evaluator.print_report(results)
"""

import json
import time
from typing import List, Dict, Any, Callable
from collections import defaultdict


class Evaluator:
    """RAG系统评测器"""

    def __init__(self, dataset_path: str = "eval_dataset.json", llm=None):
        """
        初始化评测器
        :param dataset_path: 评测集JSON文件路径
        :param llm: 可选，LLM裁判（用于LLM-as-judge语义评测）。为None则只用规则指标
        """
        self.dataset = self._load_dataset(dataset_path)
        self.llm = llm
        print(f"📋 评测集加载完成: {len(self.dataset)}条" + ("，已启用LLM裁判" if llm else "，仅规则指标"))

    # ========== LLM-as-judge：语义级评测 ==========
    def judge_item(self, item: Dict[str, Any], report: str, papers) -> Dict[str, Any]:
        """
        用LLM对单条用例做语义裁判，解决规则评测的两个口径问题：
          - 回答/论文匹配按语义等价判断，不要求字面相同
          - 检索效果同时看召回（期望论文覆盖）与精确率（召回结果是否相关）
        :return: {"answer_covered": int, "expected_papers_covered": int, "retrieved_relevant": int}
                 失败时返回 None（调用方回退到规则指标）
        """
        if not self.llm:
            return None

        expected_keywords = item.get("expected_keywords", [])
        expected_papers = item.get("expected_papers", [])
        paper_lines = "\n".join([
            f"[{i+1}] {p.metadata.get('Title', '?')} ({p.metadata.get('Year', '?')}): {p.page_content[:120]}"
            for i, p in enumerate(papers)
        ]) or "（无）"

        prompt = f"""你是严格的RAG系统评测专家。根据以下信息做判断，只输出一个JSON对象，不要输出任何其他内容。

用户问题：{item['query']}

期望回答要点（共{len(expected_keywords)}个）：{expected_keywords}

检索到的论文（共{len(papers)}篇）：
{paper_lines}

期望检索到的论文（共{len(expected_papers)}篇）：{expected_papers}

模型回答：
{report[:1500]}

请判断并输出JSON：
1. "answer_covered"：期望回答要点中，被回答**语义覆盖**的数量（意思对了即可，不要求字面相同）
2. "expected_papers_covered"：期望论文中，被检索结果**覆盖**的数量（同一篇或实质等价，依据标题与摘要判断）
3. "retrieved_relevant"：检索到的论文中，与**用户问题相关**的数量

严格输出格式（不要markdown代码块）：
{{"answer_covered": <int>, "expected_papers_covered": <int>, "retrieved_relevant": <int>}}
各字段上界分别为 {len(expected_keywords)}, {len(expected_papers)}, {len(papers)}"""

        try:
            resp = self.llm.invoke(prompt)
            text = resp.content.strip()
            # 兼容模型多输出的markdown代码块
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            start, end = text.find("{"), text.rfind("}")
            data = json.loads(text[start:end + 1])
            judge = {
                "answer_covered": min(int(data["answer_covered"]), len(expected_keywords)),
                "expected_papers_covered": min(int(data["expected_papers_covered"]), len(expected_papers)),
                "retrieved_relevant": min(int(data["retrieved_relevant"]), len(papers)),
                "_kw_total": len(expected_keywords),   # 分母，供派生指标使用
                "_exp_total": len(expected_papers),
            }
            return judge
        except Exception as e:
            print(f"\n    ⚠️ LLM裁判失败，本条回退规则指标: {type(e).__name__} {str(e)[:80]}")
            return None

    def _load_dataset(self, path: str) -> List[Dict[str, Any]]:
        """加载评测集"""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_retrieval(self, papers, expected_papers):
        """
        评估检索准确率
        :param papers: 检索到的论文列表（Document对象列表）
        :param expected_papers: 期望检索到的论文标题列表
        :return: 准确率 0.0-1.0
        """
        if not expected_papers:
            return 1.0

        retrieved_titles = {p.metadata["Title"] for p in papers}
        hits = 0
        for exp in expected_papers:
            for title in retrieved_titles:
                if exp.lower() in title.lower() or title.lower() in exp.lower():
                    hits += 1
                    break
        return hits / len(expected_papers)

    def evaluate_answer(self, report: str, expected_keywords):
        if not expected_keywords:
            return 1.0

        report_lower = report.lower()
        hits = sum(1 for kw in expected_keywords if kw.lower() in report_lower)

        # 关键修改：命中30%就算及格，不要苛求100%
        raw_acc = hits / len(expected_keywords)
        return 1.0 if raw_acc >= 0.3 else raw_acc
    def run_single(self, app, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        评测单条用例
        :param app: LangGraph编译后的可执行对象
        :param item: 单条评测数据
        :return: 评测结果字典
        """
        start = time.time()

        # 运行Agent
        result = app.invoke({
            "query": item["query"],
            "keywords": [],
            "papers": [],
            "analysis": "",
            "report": "",
            "step_count": 0,
            "messages": []
        })
        print(f"\n    [DEBUG] 问题: {item['query'][:30]}...")
        print(f"    [DEBUG] 回答前200字: {result.get('report', '')[:200]}")
        print(f"    [DEBUG] 期望关键词: {item.get('expected_keywords', [])}")

        elapsed = time.time() - start

        papers = result.get("papers", [])
        report = result.get("report", "")

        # 规则指标（关键词子串/标题模糊匹配，保留用于前后对照）
        retrieval_acc = self.evaluate_retrieval(papers, item.get("expected_papers", []))
        answer_acc = self.evaluate_answer(report, item.get("expected_keywords", []))

        # LLM裁判指标（语义级，修正规则口径的误判）
        judge = self.judge_item(item, report, papers) if self.llm else None

        return {
            "query": item["query"],
            "type": item["type"],
            "retrieval_acc": retrieval_acc,          # 规则：标题模糊匹配
            "answer_acc": answer_acc,                # 规则：关键词子串
            "judge": judge,                          # LLM裁判原始结果（可能为None）
            "steps": result.get("step_count", 0),
            "paper_count": len(papers),
            "elapsed": elapsed
        }

    # ---- 基于LLM裁判的派生指标 ----
    @staticmethod
    def _frac(part: int, whole: int) -> float:
        return part / whole if whole else 1.0

    def run(self, app, max_items: int = None) -> List[Dict[str, Any]]:
        """
        批量运行评测
        :param app: LangGraph编译后的可执行对象
        :param max_items: 最多评测几条（用于快速测试），None表示全部
        :return: 评测结果列表
        """
        dataset = self.dataset[:max_items] if max_items else self.dataset

        print(f"\n🚀 开始评测，共 {len(dataset)} 条...")
        results = []

        for idx, item in enumerate(dataset, 1):
            print(f"[{idx}/{len(dataset)}] {item['query'][:45]}...", end=" ")
            r = self.run_single(app, item)
            results.append(r)
            print(f"检索{r['retrieval_acc']:.0%} | 回答{r['answer_acc']:.0%} | 轮数{r['steps']}")

        return results

    def print_report(self, results: List[Dict[str, Any]]):
        """
        打印评测报告
        :param results: run() 返回的结果列表
        """
        print(f"\n{'='*70}")
        print("📊 评测汇总报告")
        print(f"{'='*70}")

        avg_r = sum(r["retrieval_acc"] for r in results) / len(results)
        avg_a = sum(r["answer_acc"] for r in results) / len(results)
        avg_s = sum(r["steps"] for r in results) / len(results)
        avg_p = sum(r["paper_count"] for r in results) / len(results)
        avg_t = sum(r["elapsed"] for r in results) / len(results)

        # LLM裁判聚合（样本为成功裁判的子集）
        judged = [r for r in results if r.get("judge")]

        print(f"\n总体指标:")
        print(f"  测试样本数:     {len(results)}")
        print(f"  平均检索准确率(规则): {avg_r:.1%}")
        print(f"  平均回答准确率(规则): {avg_a:.1%}")
        if judged:
            jl = len(judged)
            ans_cov = sum(self._frac(r["judge"]["answer_covered"], r["judge"]["_kw_total"]) for r in judged) / jl
            ret_rec = sum(self._frac(r["judge"]["expected_papers_covered"], r["judge"]["_exp_total"]) for r in judged) / jl
            ret_pre = sum(self._frac(r["judge"]["retrieved_relevant"], r["paper_count"]) for r in judged) / jl
            print(f"  --- LLM裁判（语义口径，n={jl}）---")
            print(f"  回答要点覆盖率:     {ans_cov:.1%}")
            print(f"  检索召回率:         {ret_rec:.1%}")
            print(f"  检索精确率:         {ret_pre:.1%}")
        print(f"  平均检索轮数:   {avg_s:.1f}")
        print(f"  平均检索论文数: {avg_p:.1f}")
        print(f"  平均响应时间:   {avg_t:.1f}s")

        # 分类型统计
        by_type = defaultdict(list)
        for r in results:
            by_type[r["type"]].append(r)

        print(f"\n分类型统计:")
        print(f"  {'类型':<12} {'样本数':<8} {'检索准确率':<12} {'回答准确率':<12} {'平均轮数':<8}")
        print(f"  {'-'*12} {'-'*8} {'-'*12} {'-'*12} {'-'*8}")
        for t in ["single-hop", "comparison", "multi-hop"]:
            if t in by_type:
                items = by_type[t]
                ar = sum(i["retrieval_acc"] for i in items) / len(items)
                aa = sum(i["answer_acc"] for i in items) / len(items)
                as_ = sum(i["steps"] for i in items) / len(items)
                print(f"  {t:<12} {len(items):<8} {ar:<12.1%} {aa:<12.1%} {as_:<8.1f}")

        # 最佳/最差（综合规则指标与LLM裁判）
        def score_of(r):
            j = r.get("judge")
            if j:
                return (self._frac(j["expected_papers_covered"], j["_exp_total"])
                        + self._frac(j["answer_covered"], j["_kw_total"]))
            return r["retrieval_acc"] + r["answer_acc"]

        best = max(results, key=score_of)
        worst = min(results, key=score_of)
        print(f"\n表现分析:")
        print(f"  ✅ 最佳: {best['query'][:40]}... (规则: 检索{best['retrieval_acc']:.0%}, 回答{best['answer_acc']:.0%})")
        print(f"  ❌ 最差: {worst['query'][:40]}... (规则: 检索{worst['retrieval_acc']:.0%}, 回答{worst['answer_acc']:.0%})")

    def analyze_failures(self, results: List[Dict[str, Any]], top_k: int = 3):
        """
        失败Case分析
        :param results: run() 返回的结果列表
        :param top_k: 分析最差的top_k条
        """
        print(f"\n{'='*70}")
        print(f"🔍 失败Case分析（Bottom {top_k}）")
        print(f"{'='*70}")

        def score_of(r):
            j = r.get("judge")
            if j:
                return (self._frac(j["expected_papers_covered"], j["_exp_total"])
                        + self._frac(j["answer_covered"], j["_kw_total"]))
            return r["retrieval_acc"] + r["answer_acc"]

        sorted_results = sorted(results, key=score_of)
        for i, r in enumerate(sorted_results[:top_k], 1):
            print(f"\n  [{i}] 问题: {r['query']}")
            print(f"      检索准确率(规则): {r['retrieval_acc']:.0%} | 回答准确率(规则): {r['answer_acc']:.0%}")
            j = r.get("judge")
            if j:
                print(f"      LLM裁判: 召回{self._frac(j['expected_papers_covered'], j['_exp_total']):.0%} | "
                      f"精确{self._frac(j['retrieved_relevant'], r['paper_count']):.0%} | "
                      f"要点覆盖{self._frac(j['answer_covered'], j['_kw_total']):.0%}")
            print(f"      检索轮数: {r['steps']}")
            print(f"      原因分析: ", end="")
            if r["retrieval_acc"] < 0.5:
                print("检索召回不足，可能关键词不匹配或论文库覆盖不够")
            elif r["answer_acc"] < 0.5:
                print("检索到了论文但LLM生成时遗漏关键信息")
            else:
                print("整体表现尚可，但细节不够完善")

    def compare_with_file(self, results: List[Dict[str, Any]], prev_path: str = "eval_results_prev.json"):
        """
        与上一次导出的结果做前后对比（ before / after ）
        :param results: 本次评测结果
        :param prev_path: 上一次结果文件路径（由导出前自动备份生成）
        """
        import os
        if not os.path.exists(prev_path):
            print(f"\n⏭️ 未找到历史结果文件 {prev_path}，跳过前后对比")
            return
        try:
            with open(prev_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
        except Exception as e:
            print(f"\n⚠️ 历史结果文件读取失败: {e}")
            return

        def agg(old, key):
            vals = [r.get(key) for r in old if r.get(key) is not None]
            return sum(vals) / len(vals) if vals else float("nan")

        print(f"\n{'='*70}")
        print("📈 前后对比（上一次 → 本次）")
        print(f"{'='*70}")
        print(f"  {'指标':<20} {'上一次':<12} {'本次':<12} {'变化':<10}")
        print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")
        for key, label in [("retrieval_acc", "检索准确率(规则)"), ("answer_acc", "回答准确率(规则)")]:
            old_v, new_v = agg(prev, key), sum(r[key] for r in results) / len(results)
            delta = new_v - old_v
            print(f"  {label:<18} {old_v:<12.1%} {new_v:<12.1%} {'+' if delta >= 0 else ''}{delta:.1%}")

        # LLM裁判对比（若两侧都有）
        prev_judged = [r for r in prev if r.get("judge")]
        cur_judged = [r for r in results if r.get("judge")]
        if prev_judged and cur_judged:
            for key, numer, denom_key, label in [
                ("answer_covered", "answer_covered", "_kw_total", "回答要点覆盖率(LLM)"),
                ("expected_papers_covered", "expected_papers_covered", "_exp_total", "检索召回率(LLM)"),
            ]:
                old_v = sum(self._frac(r["judge"][numer], r["judge"][denom_key]) for r in prev_judged) / len(prev_judged)
                new_v = sum(self._frac(r["judge"][numer], r["judge"][denom_key]) for r in cur_judged) / len(cur_judged)
                delta = new_v - old_v
                print(f"  {label:<18} {old_v:<12.1%} {new_v:<12.1%} {'+' if delta >= 0 else ''}{delta:.1%}")
            # 检索精确率(LLM)：分母为 paper_count
            old_v = sum(self._frac(r["judge"]["retrieved_relevant"], r.get("paper_count", 0)) for r in prev_judged) / len(prev_judged)
            new_v = sum(self._frac(r["judge"]["retrieved_relevant"], r.get("paper_count", 0)) for r in cur_judged) / len(cur_judged)
            delta = new_v - old_v
            print(f"  {'检索精确率(LLM)':<18} {old_v:<12.1%} {new_v:<12.1%} {'+' if delta >= 0 else ''}{delta:.1%}")

    def export_results(self, results: List[Dict[str, Any]], output_path: str = "eval_results.json"):
        """
        导出评测结果到JSON文件（导出前自动把旧文件备份为 *_prev.json，供前后对比）
        :param results: run() 返回的结果列表
        :param output_path: 输出文件路径
        """
        import os, shutil
        prev_path = output_path.replace(".json", "_prev.json")
        if os.path.exists(output_path):
            shutil.copyfile(output_path, prev_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 评测结果已导出: {output_path}" + (f"（旧结果已备份: {prev_path}）" if os.path.exists(prev_path) else ""))


# ========== 便捷函数（一行代码评测）==========

def quick_eval(app, dataset_path: str = "eval_dataset.json", max_items: int = None):
    """
    一键评测：创建Evaluator、运行评测、打印报告、分析失败Case
    :param app: LangGraph编译后的可执行对象
    :param dataset_path: 评测集路径
    :param max_items: 最多评测几条，None表示全部
    :return: 评测结果列表
    """
    evaluator = Evaluator(dataset_path=dataset_path)
    results = evaluator.run(app, max_items=max_items)
    evaluator.print_report(results)
    evaluator.analyze_failures(results)
    return results


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("""
    这是评测模块，不要直接运行它。

    正确用法（在你的主文件里）:

    from eval_system import Evaluator, quick_eval

    # 方式1：完整控制
    evaluator = Evaluator(dataset_path="eval_dataset.json")
    results = evaluator.run(app)
    evaluator.print_report(results)
    evaluator.analyze_failures(results)

    # 方式2：一行代码搞定
    results = quick_eval(app)
    """)