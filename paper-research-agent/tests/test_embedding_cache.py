"""Embedding磁盘缓存单元测试（用假后端，不调用真实API）"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# agentic_rag_hybrid 导入时要求存在 DASHSCOPE_API_KEY，这里给个假的（不会真的发请求）
os.environ.setdefault("DASHSCOPE_API_KEY", "test-dummy-key")

import agentic_rag_hybrid as rag


class FakeBackend:
    """记录调用次数的假embedding后端"""
    def __init__(self):
        self.doc_calls = 0
        self.query_calls = 0

    def embed_documents(self, texts):
        self.doc_calls += 1
        return [[float(len(t)), 0.0] for t in texts]

    def embed_query(self, text):
        self.query_calls += 1
        return [float(len(text)), 0.0]


@pytest.fixture
def cached(tmp_path, monkeypatch):
    monkeypatch.setattr(rag.DiskCachedEmbeddings, "CACHE_FILE", str(tmp_path / "embed_cache.json"))
    backend = FakeBackend()
    return rag.DiskCachedEmbeddings(backend), backend


class TestDiskCachedEmbeddings:
    def test_miss_then_hit(self, cached):
        emb, backend = cached
        v1 = emb.embed_query("hello")
        assert backend.query_calls == 1
        v2 = emb.embed_query("hello")
        assert backend.query_calls == 1  # 缓存命中，不再调用后端
        assert v1 == v2

    def test_batch_documents_partial_cache(self, cached):
        emb, backend = cached
        emb.embed_documents(["a", "bb"])          # 2个miss
        emb.embed_documents(["a", "bb", "ccc"])   # 前2个hit，1个miss
        assert backend.doc_calls == 2

    def test_persistence_across_instances(self, cached, tmp_path, monkeypatch):
        emb, backend = cached
        emb.embed_query("persist me")
        assert backend.query_calls == 1

        backend2 = FakeBackend()
        emb2 = rag.DiskCachedEmbeddings(backend2)
        emb2.embed_query("persist me")
        assert backend2.query_calls == 0  # 从磁盘缓存恢复，零后端调用
