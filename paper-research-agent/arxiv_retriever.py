"""
arxiv_retriever.py — 真实 arXiv 检索模块
通过 urllib 直接调用 arXiv 官方 API（不依赖易损坏的第三方 arxiv 包），
返回 LangChain Document 列表，并带本地 JSON 缓存，避免重复请求。

用法:
    from arxiv_retriever import search_arxiv

    papers = search_arxiv(["sliding window attention", "Longformer"], max_results=8)
"""

import os
import json
import time
import ssl
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List
from langchain_core.documents import Document

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arxiv_cache.json")
CACHE_TTL = 7 * 24 * 3600  # 缓存7天
REQUEST_TIMEOUT = 20       # 单次请求超时（秒）
REQUEST_SLEEP = 1.0        # 缓存未命中时，请求间隔（秒，礼貌访问）

ATOM_NS = "{http://www.w3.org/2005/Atom}"


def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict):
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 缓存写入失败: {e}")


def _http_get(url: str) -> bytes:
    """GET 请求：优先正常验证证书，证书链验证失败时降级为不验证（兼容本机证书链问题）"""
    req = urllib.request.Request(url, headers={"User-Agent": "paper-research-agent/1.0"})
    try:
        return urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT).read()
    except urllib.error.URLError as e:
        # urlopen 会把 SSLError 包在 URLError.reason 里，需要拆开判断
        if not isinstance(e.reason, ssl.SSLError):
            raise
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT, context=ctx).read()


def _parse_feed(xml_data: bytes) -> List[Document]:
    """解析 arXiv Atom 响应为 Document 列表"""
    root = ET.fromstring(xml_data)
    papers = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        title = " ".join(entry.find(f"{ATOM_NS}title").text.split())
        summary = " ".join(entry.find(f"{ATOM_NS}summary").text.split())
        published = entry.find(f"{ATOM_NS}published").text  # 形如 2017-06-12T17:57:34Z
        link = entry.find(f"{ATOM_NS}id").text
        arxiv_id = link.rsplit("/abs/", 1)[-1] if "/abs/" in link else link
        papers.append(Document(
            page_content=f"{title}. {summary}",
            metadata={
                "Title": title,
                "Year": published[:4],
                "arxiv_id": arxiv_id,
                "url": link,
                "source": "arxiv"
            }
        ))
    return papers


def _build_query(keywords: List[str]) -> str:
    """关键词 → arXiv search_query（每个词限定 all 字段并加引号）"""
    terms = [f'all:"{k.strip()}"' for k in keywords if k and k.strip()]
    return " AND ".join(terms) if terms else ""


def _do_search(query: str, max_results: int) -> List[Document]:
    params = urllib.parse.urlencode({
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance"
    })
    url = f"http://export.arxiv.org/api/query?{params}"
    return _parse_feed(_http_get(url))


def search_arxiv(keywords: List[str], max_results: int = 8,
                 use_cache: bool = True, verbose: bool = True) -> List[Document]:
    """
    按关键词检索 arXiv，返回 Document 列表（失败时返回空列表，由调用方降级处理）。

    :param keywords: 检索关键词列表
    :param max_results: 最多返回多少篇
    :param use_cache: 是否使用本地缓存
    :param verbose: 是否打印检索日志
    """
    keywords = [k for k in (keywords or []) if k and k.strip()]
    if not keywords:
        return []

    cache_key = f"{'|'.join(keywords)}#{max_results}"
    if use_cache:
        cache = _load_cache()
        hit = cache.get(cache_key)
        if hit and time.time() - hit["time"] < CACHE_TTL:
            if verbose:
                print(f"🌐 arXiv(缓存): {len(hit['papers'])} 篇")
            return [Document(page_content=p["content"], metadata=p["metadata"])
                    for p in hit["papers"]]

    query_and = _build_query(keywords)
    attempts = [query_and, " OR ".join(query_and.split(" AND "))]
    if verbose:
        print(f"🌐 arXiv检索: {keywords} (最多{max_results}篇)")

    papers = []
    for q in attempts:
        if not q:
            continue
        try:
            time.sleep(REQUEST_SLEEP)  # 礼貌访问，避免触发限流
            results = _do_search(q, max_results)
        except Exception as e:
            if verbose:
                print(f"⚠️ arXiv请求失败: {type(e).__name__} {str(e)[:120]}")
            return []  # 网络问题直接降级，交给本地语料
        # 候选多多益善（下游还有向量+关键词混合重排序），取结果更多的一次
        if len(results) > len(papers):
            papers = results
        if len(papers) >= max_results:
            break

    if verbose:
        print(f"🌐 arXiv返回: {len(papers)} 篇")

    if use_cache and papers:
        cache = _load_cache()
        cache[cache_key] = {
            "time": time.time(),
            "papers": [{"content": p.page_content, "metadata": p.metadata} for p in papers]
        }
        _save_cache(cache)

    return papers
