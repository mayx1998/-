import os
import json
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings
from memory_system import LongTermMemory
from arxiv_retriever import search_arxiv


# 全局变量（在文件顶部初始化）
long_term_memory = None
EVAL_MODE = False  # 评测模式：隔离长期记忆，防止记忆漂移污染检索指标


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
    model="text-embedding-v4",  # v3免费额度已耗尽，换用v4（同样有免费额度）
    dashscope_api_key=API_KEY
)


# ========== Embedding磁盘缓存：同一文本只调一次API，节省免费额度 ==========
import hashlib


class DiskCachedEmbeddings:
    """给任意embedding后端加一层持久化缓存（按文本MD5存json）"""
    CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embed_cache.json")

    def __init__(self, backend):
        self.backend = backend
        self.cache = {}
        if os.path.exists(self.CACHE_FILE):
            try:
                with open(self.CACHE_FILE, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def _key(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _save(self):
        with open(self.CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(self.cache, f)

    def embed_documents(self, texts):
        results = [None] * len(texts)
        misses = []
        for i, t in enumerate(texts):
            k = self._key(t)
            if k in self.cache:
                results[i] = self.cache[k]
            else:
                misses.append((i, t, k))
        if misses:
            vecs = self.backend.embed_documents([t for _, t, _ in misses])
            for (i, _, k), v in zip(misses, vecs):
                self.cache[k] = v
                results[i] = v
            self._save()
        return results

    def embed_query(self, text):
        k = self._key(text)
        if k not in self.cache:
            self.cache[k] = self.backend.embed_query(text)
            self._save()
        return self.cache[k]


embeddings = DiskCachedEmbeddings(embeddings)


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
    messages: List[dict]

# ========== 节点1：分析Query，生成检索关键词 ==========
def analyze_query(state: AgentState):
    # 获取长期记忆上下文（评测模式下关闭：记忆会干扰关键词提取的客观性）
    memory_context = ""
    if long_term_memory and not EVAL_MODE:
        memory_context = long_term_memory.get_context_prompt()

    prompt = f"""你是一个学术研究助手。{memory_context}

请分析用户的问题，提取3个最相关的检索关键词。
用户问题：{state['query']}

要求：
1. 关键词应该是英文
2. 每行一个关键词，不要编号，不要解释
3. 如果用户问题涉及之前聊过的话题，请结合上下文理解

示例输出：
attention mechanism
transformer architecture
self-attention"""

    response = llm.invoke(prompt)
    keywords = [k.strip() for k in response.content.strip().split('\n') if k.strip()]
    state["keywords"] = keywords[:3]
    state["step_count"] = 0
    state["messages"] = state.get("messages", [])  # 确保messages存在
    print(f"🔍 生成关键词: {state['keywords']}")
    return state

# ========== 多轮检索：用LLM改写关键词，避免重复同一查询导致空转 ==========
def rewrite_keywords(state: AgentState) -> List[str]:
    """第2轮起调用：基于原问题和已用过的关键词，生成不同角度的检索词"""
    prev_kws = state.get("keywords", [])
    prompt = f"""用户问题：{state['query']}
之前用过的检索关键词：{', '.join(prev_kws)}
这些关键词没有检索到足够的信息。请从**不同角度**给出3个新的英文检索关键词
（可以更宽泛、用同义词、或换成问题中的核心实体），每行一个，不要编号、不要解释，
不要与已有关键词重复。"""
    try:
        response = llm.invoke(prompt)
        kws = [k.strip() for k in response.content.strip().split('\n') if k.strip()]
        prev_lower = {k.lower() for k in prev_kws}
        new_kws = [k for k in kws if k.lower() not in prev_lower]
        return new_kws[:3]
    except Exception as e:
        print(f"⚠️ 关键词改写失败: {e}")
        return []


# ========== 节点2：检索论文（arXiv实时检索 + 本地语料，混合排序）==========
def retrieve_papers_hybrid(state: AgentState):
    keywords = state.get("keywords", [])
    if not keywords:
        keywords = [state["query"]]

    # 加入长期记忆兴趣作为辅助关键词（评测模式下关闭，防止记忆漂移污染指标）
    if long_term_memory and not EVAL_MODE:
        interest_kws = long_term_memory.get_interests(2)
        if interest_kws:
            keywords = keywords + interest_kws
            print(f"🧠 记忆辅助: {interest_kws}")

    # 第2轮起：改写关键词，换个角度检索（否则同一查询只会召回相同论文，空转3轮）
    if state.get("step_count", 0) >= 1:
        new_kws = rewrite_keywords(state)
        if new_kws:
            keywords = new_kws
            print(f"🔁 第{state['step_count']+1}轮关键词改写: {new_kws}")

    # ========== 真实 arXiv 检索：按轮次扩大范围，保证后续轮次能补充新论文 ==========
    # 第1轮拉4篇，第2轮拉8篇，第3轮拉12篇（去重后真正新增）
    max_results = 4 * (state.get("step_count", 0) + 1)
    arxiv_papers = search_arxiv(keywords[:3], max_results=max_results)  # 主关键词检索，避免记忆兴趣带偏
    corpus = LOCAL_PAPERS + arxiv_papers  # 本地语料作为保底/补充

    # ========== 第1路：向量检索（语义匹配）==========
    # 先把所有论文向量化
    paper_texts = [p.page_content for p in corpus]
    paper_embeddings = embeddings.embed_documents(paper_texts)

    # Query向量化：直接用用户原始问题（而非拼接关键词）
    # 原始问题语义完整，多语言embedding(v3)能更好匹配论文摘要
    query_vec = embeddings.embed_query(state["query"])

    # 计算余弦相似度
    def cosine_sim(a, b):
        import numpy as np
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    vec_scores = [(i, cosine_sim(query_vec, emb)) for i, emb in enumerate(paper_embeddings)]
    vec_scores.sort(key=lambda x: x[1], reverse=True)
    vec_results = [corpus[i] for i, _ in vec_scores]  # 按相似度排序的论文

    # ========== 第2路：关键词检索（精确匹配）==========
    kw_scores = {}
    for i, paper in enumerate(corpus):
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
    kw_results = [corpus[i] for i, _ in kw_sorted]

    # ========== 融合：RRF（倒数排名融合）==========
    # RRF公式: score = Σ 1/(k + rank)，k通常取60
    K = 60
    rrf_scores = {}

    # 向量检索的排名贡献
    for rank, (i, _) in enumerate(vec_scores):
        paper_id = corpus[i].metadata["Title"]
        rrf_scores[paper_id] = rrf_scores.get(paper_id, 0) + 1 / (K + rank + 1)

    # 关键词检索的排名贡献
    for rank, (i, _) in enumerate(kw_sorted):
        paper_id = corpus[i].metadata["Title"]
        rrf_scores[paper_id] = rrf_scores.get(paper_id, 0) + 1 / (K + rank + 1)

    # 按RRF分数排序
    final_sorted = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # 取Top结果
    title_to_paper = {p.metadata["Title"]: p for p in corpus}
    retrieved = [title_to_paper[title] for title, _ in final_sorted[:6]]  # Top-6：给多主题问题足够覆盖度

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

    # 更新短期记忆
    state["messages"] = state.get("messages", []) + [
        {"role": "user", "content": state["query"]},
        {"role": "assistant", "content": state["report"]}
    ]
    if len(state["messages"]) > 10:  # Sliding Window：只保留最近5轮
        state["messages"] = state["messages"][-10:]

    # 更新长期记忆（评测模式下关闭写入，避免评测数据污染用户画像）
    if long_term_memory and not EVAL_MODE:
        new_interests = long_term_memory.update(state["query"], state["report"])
        print(f"🧠 长期记忆更新: {new_interests}")

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
    # 评测模式：隔离长期记忆的注入与写入，排除记忆漂移对检索指标的干扰
    EVAL_MODE = True

    # 初始化长期记忆（保留，但评测期间不生效）
    long_term_memory = LongTermMemory(llm=llm)
    print(f"🧠 长期记忆已加载(评测模式，已隔离): {long_term_memory.get_stats()}")

    # ========== 调用独立评测模块（规则指标 + LLM-as-judge）==========
    from eval_system import Evaluator

    # 裁判LLM：temperature=0保证打分稳定可复现
    judge_llm = ChatOpenAI(
        model="qwen-plus",
        api_key=API_KEY,
        base_url=API_BASE,
        temperature=0,
        timeout=60
    )

    evaluator = Evaluator(dataset_path="eval_dataset.json", llm=judge_llm)
    results = evaluator.run(app)
    evaluator.print_report(results)
    evaluator.analyze_failures(results)
    evaluator.export_results(results)        # 导出 eval_results.json（旧结果自动备份为 _prev）
    evaluator.compare_with_file(results)     # 与上一次结果做前后对比
