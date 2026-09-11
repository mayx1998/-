"""记忆系统单元测试（离线，无需API key）"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_system import ShortTermMemory, LongTermMemory


@pytest.fixture
def ltm(tmp_path):
    """使用临时记忆文件，避免污染真实的 user_memory.json"""
    return LongTermMemory(llm=None, memory_file=str(tmp_path / "memory.json"))


class TestShortTermMemory:
    def test_add_turn_and_window(self):
        stm = ShortTermMemory(max_history=4)  # 最多2轮
        stm.add_turn("q1", "a1")
        stm.add_turn("q2", "a2")
        stm.add_turn("q3", "a3")  # 超出窗口，最早的被丢弃
        assert len(stm.messages) == 4
        assert stm.messages[0]["content"] == "q2"

    def test_get_context_empty(self):
        stm = ShortTermMemory()
        assert stm.get_context() == ""

    def test_roundtrip_list(self):
        stm = ShortTermMemory()
        stm.add_turn("q", "a")
        stm2 = ShortTermMemory()
        stm2.from_list(stm.to_list())
        assert stm2.messages == stm.messages


class TestLongTermMemory:
    def test_rule_extract_interests(self, ltm):
        # 无LLM时走规则提取：取query中的大写专有名词
        interests = ltm.extract_interests("What is BERT and GPT?", "answer")
        assert "BERT" in interests and "GPT" in interests

    def test_update_and_persist(self, ltm, tmp_path):
        ltm.update("介绍一下 Transformer", "Transformer基于Attention")
        # 重新加载，验证持久化
        ltm2 = LongTermMemory(llm=None, memory_file=str(tmp_path / "memory.json"))
        stats = ltm2.get_stats()
        assert stats["session_count"] == 1
        assert stats["total_topics"] == 1

    def test_interest_dedup_and_cap(self, ltm):
        for i in range(30):
            ltm.data["interests"].append(f"topic_{i}")
        ltm.update("What is GPT?", "answer")  # 规则提取出 GPT
        interests = ltm.data["interests"]
        assert len(interests) <= 20  # MAX_INTERESTS 上限
        assert "GPT" in interests

    def test_get_context_prompt_empty(self, ltm):
        assert ltm.get_context_prompt() == ""

    def test_clear(self, ltm):
        ltm.update("What is BERT?", "answer")
        ltm.clear()
        assert ltm.get_stats()["total_interests"] == 0
