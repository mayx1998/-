# 智能论文检索系统（Agentic RAG）

基于 LangGraph 的多轮智能论文检索问答系统：arXiv 实时检索 + 混合检索（RRF 融合）+ LLM 自评估循环 + 用户兴趣记忆 + 规则指标/LLM-as-judge 双重评测闭环 + FastAPI/SSE 流式 Web 界面。

**完整文档（架构图 · 快速开始 · 评测结果 · 消融实验）见 👉 [paper-research-agent/README.md](paper-research-agent/README.md)**

## 核心指标（20 条评测集）

| 指标 | 规则口径 | LLM-as-judge |
|---|---|---|
| 检索准确率 | 81.0%（优化自 52.5%） | 召回 81.0% / 精确率 31.1% |
| 回答准确率 | 96.2% | 要点覆盖率 83.6% |

## 快速体验

```bash
cd paper-research-agent
pip install -r requirements.txt
export DASHSCOPE_API_KEY=你的key   # Windows: $env:DASHSCOPE_API_KEY="..."
python api.py                       # 打开 http://localhost:8000
python -m pytest tests -v           # 28个离线单测
```
