"""
api.py — 论文检索系统的 FastAPI 服务入口
提供:
  - GET  /        : 内置聊天页面
  - POST /query   : SSE 流式返回检索过程 + 最终报告 + 引用论文链接

启动:
    python api.py
    或: uvicorn api:app --reload --port 8000
然后浏览器打开 http://localhost:8000
"""

import json
import agentic_rag_hybrid as rag
from memory_system import LongTermMemory
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# 初始化长期记忆（与主程序的 __main__ 逻辑一致）
rag.long_term_memory = LongTermMemory(llm=rag.llm)

app = FastAPI(title="智能论文检索系统", version="1.0")


class QueryRequest(BaseModel):
    query: str


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _initial_state(query: str) -> dict:
    return {
        "query": query,
        "keywords": [],
        "papers": [],
        "analysis": "",
        "report": "",
        "step_count": 0,
        "messages": []
    }


def _paper_info(p) -> dict:
    return {
        "title": p.metadata.get("Title", "未知标题"),
        "year": p.metadata.get("Year", ""),
        "url": p.metadata.get("url", ""),
        "source": p.metadata.get("source", "local"),
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return CHAT_PAGE


@app.post("/query")
def query(req: QueryRequest):
    def event_stream():
        final_state = None
        try:
            yield _sse({"type": "start", "query": req.query})
            # 逐节点流式输出 Agent 的执行过程
            for chunk in rag.app.stream(_initial_state(req.query)):
                node, state = next(iter(chunk.items()))
                final_state = state
                yield _sse({
                    "type": "node",
                    "node": node,
                    "keywords": state.get("keywords", []),
                    "step": state.get("step_count", 0),
                    "analysis": state.get("analysis", ""),
                })
            report = (final_state or {}).get("report", "")
            papers = [_paper_info(p) for p in (final_state or {}).get("papers", [])]
            yield _sse({"type": "done", "report": report, "papers": papers})
        except Exception as e:
            yield _sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


CHAT_PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>智能论文检索系统</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: "Microsoft YaHei", sans-serif; background: #f0f2f5; height: 100vh; display: flex; flex-direction: column; }
  header { background: #1a73e8; color: #fff; padding: 14px 20px; font-size: 18px; font-weight: bold; }
  #chat { flex: 1; overflow-y: auto; padding: 16px; max-width: 860px; width: 100%; margin: 0 auto; }
  .msg { margin-bottom: 14px; display: flex; }
  .msg.user { justify-content: flex-end; }
  .bubble { max-width: 78%; padding: 10px 14px; border-radius: 12px; white-space: pre-wrap; line-height: 1.7; font-size: 14px; }
  .msg.user .bubble { background: #1a73e8; color: #fff; }
  .msg.bot .bubble { background: #fff; border: 1px solid #e3e6ea; }
  .progress { color: #6b7280; font-size: 13px; margin: 4px 0; }
  .papers { margin-top: 10px; border-top: 1px dashed #d1d5db; padding-top: 8px; }
  .papers a { color: #1a73e8; text-decoration: none; font-size: 13px; }
  .papers li { margin: 3px 0 3px 18px; }
  #input-bar { display: flex; gap: 8px; padding: 12px 16px; background: #fff; border-top: 1px solid #e3e6ea; }
  #q { flex: 1; padding: 10px 14px; border: 1px solid #d1d5db; border-radius: 20px; font-size: 14px; outline: none; }
  #q:focus { border-color: #1a73e8; }
  button { padding: 10px 22px; border: none; border-radius: 20px; background: #1a73e8; color: #fff; font-size: 14px; cursor: pointer; }
  button:disabled { background: #9ca3af; cursor: default; }
  .tag { display: inline-block; background: #e8f0fe; color: #1a73e8; border-radius: 10px; padding: 1px 8px; font-size: 12px; margin: 1px 3px 1px 0; }
</style>
</head>
<body>
<header>📚 智能论文检索系统（Agentic RAG）</header>
<div id="chat"></div>
<div id="input-bar">
  <input id="q" placeholder="输入你的论文问题，例如：Longformer如何降低Attention的复杂度？" autofocus>
  <button id="send" onclick="ask()">发送</button>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('q');
const btn = document.getElementById('send');
let busy = false;

function addMsg(cls, html) {
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.innerHTML = '<div class="bubble">' + html + '</div>';
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div.querySelector('.bubble');
}

input.addEventListener('keydown', e => { if (e.key === 'Enter') ask(); });

async function ask() {
  const query = input.value.trim();
  if (!query || busy) return;
  busy = true; btn.disabled = true; input.value = '';
  addMsg('user', query);

  const bubble = addMsg('bot', '<span class="progress">🔍 正在分析关键词...</span>');
  const progress = bubble.querySelector('.progress');
  let reportText = '';

  try {
    const resp = await fetch('/query', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query})
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += decoder.decode(value, {stream: true});
      const parts = buf.split('\\n\\n');
      buf = parts.pop();
      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith('data:')) continue;
        const ev = JSON.parse(line.slice(5));
        if (ev.type === 'node') {
          if (ev.node === 'analyze') progress.textContent = '🔍 关键词: ' + ev.keywords.join(', ');
          if (ev.node === 'retrieve') progress.textContent = '📚 第' + ev.step + '轮检索完成（arXiv + 本地混合）...';
          if (ev.node === 'evaluate') progress.textContent = '📊 评估: ' + ev.analysis;
          if (ev.node === 'generate') progress.textContent = '✍️ 正在生成报告...';
        } else if (ev.type === 'done') {
          reportText = ev.report || '（未生成报告）';
          let html = reportText.replace(/&/g,'&amp;').replace(/</g,'&lt;');
          if (ev.papers && ev.papers.length) {
            html += '<div class="papers"><b>📖 引用论文</b><ul>';
            for (const p of ev.papers) {
              const link = p.url || '#';
              html += `<li><a href="${link}" target="_blank">${p.title}</a> <span class="tag">${p.year}</span> <span class="tag">${p.source}</span></li>`;
            }
            html += '</ul></div>';
          }
          bubble.innerHTML = html;
        } else if (ev.type === 'error') {
          bubble.innerHTML = '<span style="color:red">❌ ' + ev.message + '</span>';
        }
      }
    }
  } catch (e) {
    bubble.innerHTML = '<span style="color:red">❌ 请求失败: ' + e + '</span>';
  }
  chat.scrollTop = chat.scrollHeight;
  busy = false; btn.disabled = false; input.focus();
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
