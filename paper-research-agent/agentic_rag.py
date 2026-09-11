import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document


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
def retrieve_papers(state: AgentState):
    keywords = state.get("keywords", [])
    if not keywords:
        keywords = [state["query"]]

    # 本地模拟检索：根据关键词匹配论文标题和内容
    matched = []
    for paper in LOCAL_PAPERS:
        score = 0
        text = (paper.metadata.get("Title", "") + " " + paper.page_content).lower()
        for kw in keywords:
            if kw.lower() in text:
                score += 1
        if score > 0:
            matched.append((paper, score))

    # 按匹配度排序
    matched.sort(key=lambda x: x[1], reverse=True)
    retrieved = [p for p, _ in matched[:4]]  # 取Top-4

    # 如果没有匹配到，返回所有论文（fallback）
    if not retrieved:
        retrieved = LOCAL_PAPERS[:3]

    # 合并新检索结果（避免重复）
    existing_titles = {p.metadata["Title"] for p in state.get("papers", [])}
    new_papers = [p for p in retrieved if p.metadata["Title"] not in existing_titles]

    state["papers"] = state.get("papers", []) + new_papers
    state["step_count"] += 1
    print(f"📚 第{state['step_count']}轮检索，新增 {len(new_papers)} 篇，累计 {len(state['papers'])} 篇")
    for p in state["papers"]:
        print(f"   - {p.metadata['Title']}")
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
workflow.add_node("retrieve", retrieve_papers)
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
    test_queries = [
        "Transformer中的Attention机制原理是什么？",
        "Transformer之后有哪些改进工作？",
        "BERT和GPT的预训练目标有什么区别？"
    ]

    for q in test_queries:
        print("\n" + "="*70)
        print(f"🚀 新问题: {q}")
        print("="*70)

        result = app.invoke({
            "query": q,
            "keywords": [],
            "papers": [],
            "analysis": "",
            "report": "",
            "step_count": 0
        })

        print("\n📖 最终报告:")
        print(result["report"])
        print("\n" + "="*70)