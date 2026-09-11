"""arXiv检索模块单元测试（离线：XML解析 + 查询构造，不触网）"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arxiv_retriever import _build_query, _parse_feed, ATOM_NS


SAMPLE_FEED = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <updated>2017-08-15T00:00:00Z</updated>
    <published>2017-06-12T17:57:34Z</published>
    <title>Attention Is All You Need</title>
    <summary>We propose the Transformer, based solely on attention mechanisms.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/1810.04805v2</id>
    <updated>2019-05-24T00:00:00Z</updated>
    <published>2018-10-11T01:12:28Z</published>
    <title>BERT: Pre-training of Deep
    Bidirectional Transformers</title>
    <summary>BERT uses bidirectional
    attention and two pre-training tasks.</summary>
  </entry>
</feed>""".encode("utf-8")


class TestBuildQuery:
    def test_and_join(self):
        q = _build_query(["Longformer", "sliding window"])
        assert q == 'all:"Longformer" AND all:"sliding window"'

    def test_skip_empty(self):
        q = _build_query(["BERT", " ", ""])
        assert q == 'all:"BERT"'


class TestParseFeed:
    def test_basic_fields(self):
        papers = _parse_feed(SAMPLE_FEED)
        assert len(papers) == 2
        p = papers[0]
        assert p.metadata["Title"] == "Attention Is All You Need"  # 换行被规整
        assert p.metadata["Year"] == "2017"
        assert p.metadata["arxiv_id"] == "1706.03762v7"
        assert p.metadata["url"] == "http://arxiv.org/abs/1706.03762v7"
        assert p.metadata["source"] == "arxiv"
        assert "Transformer" in p.page_content

    def test_multiline_title_normalized(self):
        papers = _parse_feed(SAMPLE_FEED)
        assert "\n" not in papers[1].metadata["Title"]

    def test_empty_feed(self):
        empty = f'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'.encode()
        assert _parse_feed(empty) == []
