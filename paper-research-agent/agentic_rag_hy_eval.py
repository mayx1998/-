import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings



# ========== 修复1：LLM改成阿里云百炼（国内直连）==========
API_KEY = os.getenv("DASHSCOPE_API_KEY")
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

if not API_KEY:
    raise ValueError("❌ 没有找到 DASHSCOPE_API_KEY 环境变量！")

llm = ChatOpenAI(
    model="qwen-plus",
    api_key=API_KEY,
    base_url=API_BASE,
    temperature=0.7,
    timeout=60  # 设置60秒超时，防止网络波动
)

embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=API_KEY
)


# ========== 修复2：本地模拟论文数据（绕过arXiv网络问题）==========
# 实际项目中，这里应该是 ArxivLoader 或 PDF 加载器
LOCAL_PAPERS = [
    Document(
        page_content="""We propose the Transformer, based solely on attention mechanisms. 
        An attention function maps a query and key-value pairs to an output. 
        The output is a weighted sum of values, where weights come from a compatibility function 
        of the query with the corresponding key. We use Scaled Dot-Product Attention.""",
        metadata={"Title": "Attention Is All You Need", "Year": "2017", "source": "arxiv"}
    ),
    Document(
        page_content="""BERT uses bidirectional attention and two pre-training tasks: 
        Masked Language Model (MLM) and Next Sentence Prediction (NSP). 
        Unlike GPT which uses left-to-right causal attention, BERT can see both left and right context.""",
        metadata={"Title": "BERT: Pre-training of Deep Bidirectional Transformers", "Year": "2019", "source": "arxiv"}
    ),
    Document(
        page_content="""GPT uses a Transformer decoder with causal (autoregressive) masking. 
        The pre-training objective is next token prediction. This left-to-right attention limits 
        the model's ability to use future context, but makes it naturally suited for generation tasks.""",
        metadata={"Title": "Improving Language Understanding by Generative Pre-Training", "Year": "2018", "source": "arxiv"}
    ),
    Document(
        page_content="""Longformer uses a combination of sliding window attention and global attention. 
        The sliding window attention attends to a fixed-size window surrounding each token, 
        reducing complexity from O(n^2) to O(n * w). Global attention allows certain tokens 
        to attend to all other tokens and be attended by all other tokens.""",
        metadata={"Title": "Longformer: The Long-Document Transformer", "Year": "2020", "source": "arxiv"}
    ),
    Document(
        page_content="""RoPE encodes position by rotating the query and key vectors in the complex plane. 
        For a token at position m, its query vector is rotated by angle m * theta. 
        The dot product of rotated vectors naturally encodes relative position (m - n). 
        This allows the model to generalize to sequence lengths not seen during training.""",
        metadata={"Title": "RoFormer: Enhanced Transformer with Rotary Position Embedding", "Year": "2021", "source": "arxiv"}
    ),
    Document(
        page_content="""Transformer-XL introduces segment-level recurrence and relative positional encoding. 
        It reuses hidden states from previous segments, enabling the model to capture longer-term dependencies 
        beyond the fixed-length context. This is a key improvement over the original Transformer for long sequences.""",
        metadata={"Title": "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context", "Year": "2019", "source": "arxiv"}
    )
]

# ========== 状态定义（不变）==========
class AgentState(TypedDict):
    query: str
    keywords: List[str]       # 提取的检索关键词
    papers: List[Document]    # 检索到的论文（改为Document类型，更规范）
    analysis: str             # 评估结论
    report: str
    step_count: int

# ========== 节点1：分析Query，生成检索关键词 ==========
def analyze_query(state: AgentState):
    prompt = f"""你是一个学术研究助手。请分析用户的问题，提取3个最相关的检索关键词。

    用户问题：{state['query']}

    要求：
    1. 关键词应该是英文（因为论文数据库是英文的）
    2. 关键词要具体，能直接用于搜索论文
    3. 输出格式：每行一个关键词，不要编号，不要解释

    示例输出：
    attention mechanism
    transformer architecture
    self-attention"""

    response = llm.invoke(prompt)
    # 解析关键词（按行分割，过滤空行）
    keywords = [k.strip() for k in response.content.strip().split('\n') if k.strip()]
    state["keywords"] = keywords[:3]  # 最多取3个
    state["step_count"] = 0
    print(f"🔍 生成关键词: {state['keywords']}")
    return state

# ========== 节点2：检索论文（本地模拟，不依赖网络）==========
def retrieve_papers_hybrid(state: AgentState):
    keywords = state.get("keywords", [])
    if not keywords:
        keywords = [state["query"]]

    # ========== 第1路：向量检索（语义匹配）==========
    # 先把所有论文向量化（如果还没做）
    paper_texts = [p.page_content for p in LOCAL_PAPERS]
    paper_embeddings = embeddings.embed_documents(paper_texts)

    # Query向量化
    query_text = " ".join(keywords) if keywords else state["query"]
    query_vec = embeddings.embed_query(query_text)

    # 计算余弦相似度
    def cosine_sim(a, b):
        import numpy as np
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    vec_scores = [(i, cosine_sim(query_vec, emb)) for i, emb in enumerate(paper_embeddings)]
    vec_scores.sort(key=lambda x: x[1], reverse=True)
    vec_results = [LOCAL_PAPERS[i] for i, _ in vec_scores]  # 按相似度排序的论文

    # ========== 第2路：关键词检索（精确匹配）==========
    kw_scores = {}
    for i, paper in enumerate(LOCAL_PAPERS):
        score = 0
        text = (paper.metadata.get("Title", "") + " " + paper.page_content).lower()
        for kw in keywords:
            kw_lower = kw.lower()
            # 标题匹配权重更高
            if kw_lower in paper.metadata.get("Title", "").lower():
                score += 3
            # 正文匹配
            if kw_lower in text:
                score += 1
        if score > 0:
            kw_scores[i] = score

    # 按关键词匹配分数排序
    kw_sorted = sorted(kw_scores.items(), key=lambda x: x[1], reverse=True)
    kw_results = [LOCAL_PAPERS[i] for i, _ in kw_sorted]

    # ========== 融合：RRF（倒数排名融合）==========
    # RRF公式: score = Σ 1/(k + rank)，k通常取60
    K = 60
    rrf_scores = {}

    # 向量检索的排名贡献
    for rank, (i, _) in enumerate(vec_scores):
        paper_id = LOCAL_PAPERS[i].metadata["Title"]
        rrf_scores[paper_id] = rrf_scores.get(paper_id, 0) + 1 / (K + rank + 1)

    # 关键词检索的排名贡献
    for rank, (i, _) in enumerate(kw_sorted):
        paper_id = LOCAL_PAPERS[i].metadata["Title"]
        rrf_scores[paper_id] = rrf_scores.get(paper_id, 0) + 1 / (K + rank + 1)

    # 按RRF分数排序
    final_sorted = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # 取Top结果
    title_to_paper = {p.metadata["Title"]: p for p in LOCAL_PAPERS}
    retrieved = [title_to_paper[title] for title, _ in final_sorted[:4]]

    # 去重 + 合并（和之前一样）
    existing_titles = {p.metadata["Title"] for p in state.get("papers", [])}
    new_papers = [p for p in retrieved if p.metadata["Title"] not in existing_titles]

    state["papers"] = state.get("papers", []) + new_papers
    state["step_count"] += 1
    print(f"📚 第{state['step_count']}轮检索(Hybrid)，新增 {len(new_papers)} 篇")
    return state

# ========== 节点3：评估检索结果是否足够 ==========
def evaluate_results(state: AgentState):
    papers_summary = "\n".join([
        f"[{i+1}] {p.metadata['Title']} ({p.metadata['Year']}): {p.page_content[:150]}..."
        for i, p in enumerate(state["papers"])
    ])

    prompt = f"""你是一个严谨的学术评估专家。请评估当前检索到的论文是否足够回答用户问题。

    用户问题：{state['query']}

    已检索到的论文：
    {papers_summary}

    评估标准：
    - ENOUGH：论文内容能完整、准确地回答用户问题
    - NEED_MORE：信息不完整，需要检索更多相关论文

    只输出一个词：ENOUGH 或 NEED_MORE。不要解释。"""

    response = llm.invoke(prompt)
    conclusion = response.content.strip().upper()

    # 提取结论（防止LLM输出多余内容）
    if "ENOUGH" in conclusion:
        state["analysis"] = "ENOUGH"
    else:
        state["analysis"] = "NEED_MORE"

    print(f"📊 评估结果: {state['analysis']}")
    return state

# ========== 节点4：生成报告 ==========
def generate_report(state: AgentState):
    papers_text = "\n\n".join([
        f"[{i+1}] {p.metadata['Title']} ({p.metadata['Year']}):\n{p.page_content[:400]}"
        for i, p in enumerate(state["papers"])
    ])

    prompt = f"""基于以下论文片段，回答用户问题。要求：
    1. 用中文回答
    2. 每个观点标注来源编号，如"根据[1]..."
    3. 如果信息不足，明确说"现有资料无法确认"
    4. 结构化输出：先给核心结论，再展开说明

    论文片段：
    {papers_text}

    用户问题：{state['query']}

    回答："""

    response = llm.invoke(prompt)
    state["report"] = response.content
    return state

# ========== 构建图（LangGraph状态机）==========
workflow = StateGraph(AgentState)
workflow.add_node("analyze", analyze_query)
workflow.add_node("retrieve", retrieve_papers_hybrid)
workflow.add_node("evaluate", evaluate_results)
workflow.add_node("generate", generate_report)

workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "retrieve")
workflow.add_edge("retrieve", "evaluate")

# 条件边：评估结果决定下一步
def should_continue(state):
    if state["step_count"] >= 3:
        print("⛔ 达到最大检索轮数(3轮)，强制结束")
        return "generate"
    if state["analysis"] == "ENOUGH":
        print("✅ 评估通过，开始生成报告")
        return "generate"
    print("🔄 信息不足，继续检索...")
    return "retrieve"

workflow.add_conditional_edges("evaluate", should_continue, {
    "retrieve": "retrieve",
    "generate": "generate"
})
workflow.add_edge("generate", END)

app = workflow.compile()

# ========== 运行测试 ==========
if __name__ == "__main__":
    import json
    from collections import defaultdict

    # 加载评测集
    with open("eval_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)


    def evaluate_retrieval(papers, expected_papers):
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


    def evaluate_answer(report, expected_keywords):
        if not expected_keywords:
            return 1.0
        report_lower = report.lower()
        hits = sum(1 for kw in expected_keywords if kw.lower() in report_lower)
        return hits / len(expected_keywords)


    print(f"📊 开始评测，共 {len(dataset)} 条测试用例\n")
    results = []

    for idx, item in enumerate(dataset, 1):
        print(f"[{idx}/{len(dataset)}] {item['query'][:50]}...")

        result = app.invoke({
            "query": item["query"],
            "keywords": [],
            "papers": [],
            "analysis": "",
            "report": "",
            "step_count": 0
        })

        retrieval_acc = evaluate_retrieval(result.get("papers", []), item.get("expected_papers", []))
        answer_acc = evaluate_answer(result.get("report", ""), item.get("expected_keywords", []))
        steps = result.get("step_count", 0)

        results.append({
            "query": item["query"],
            "type": item["type"],
            "retrieval_acc": retrieval_acc,
            "answer_acc": answer_acc,
            "steps": steps
        })

        print(f"    检索: {retrieval_acc:.0%} | 回答: {answer_acc:.0%} | 轮数: {steps}")

    # 汇总
    print("\n" + "=" * 60)
    print("📈 评测汇总")
    print("=" * 60)

    avg_r = sum(r["retrieval_acc"] for r in results) / len(results)
    avg_a = sum(r["answer_acc"] for r in results) / len(results)
    avg_s = sum(r["steps"] for r in results) / len(results)

    print(f"  平均检索准确率: {avg_r:.2%}")
    print(f"  平均回答准确率: {avg_a:.2%}")
    print(f"  平均检索轮数:   {avg_s:.1f}")

    by_type = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r)

    print(f"\n  分类型:")
    for t in ["single-hop", "comparison", "multi-hop"]:
        if t in by_type:
            items = by_type[t]
            print(
                f"    {t}: 检索{sum(i['retrieval_acc'] for i in items) / len(items):.0%} | 回答{sum(i['answer_acc'] for i in items) / len(items):.0%} ({len(items)}条)")
