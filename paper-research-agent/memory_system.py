"""
memory_system.py — 记忆系统模块
提供短期记忆（对话历史）和长期记忆（用户兴趣画像）功能
用法: from memory_system import ShortTermMemory, LongTermMemory
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any

# ========== 长期记忆配置文件 ==========
MEMORY_FILE = "user_memory.json"
MAX_INTERESTS = 20      # 最多保留20个兴趣关键词
MAX_HISTORY = 50        # 最多保留50条历史话题


class ShortTermMemory:
    """短期记忆：维护当前会话的对话历史"""

    def __init__(self, max_history: int = 10):
        """
        初始化短期记忆
        :param max_history: 最多保留的消息条数（用户+助手各算1条），默认10条=5轮对话
        """
        self.messages: List[Dict[str, str]] = []
        self.max_history = max_history

    def add_turn(self, query: str, report: str):
        """
        添加一轮对话到记忆
        :param query: 用户问题
        :param report: 助手回答
        """
        self.messages.append({"role": "user", "content": query})
        self.messages.append({"role": "assistant", "content": report})

        # Sliding Window：超出限制时丢弃最早的
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

    def get_context(self, max_chars: int = 800) -> str:
        """
        获取格式化的对话历史，用于Prompt上下文
        :param max_chars: 最大字符数，防止过长
        :return: 格式化后的对话历史字符串
        """
        if not self.messages:
            return ""

        lines = ["对话历史："]
        for msg in self.messages:
            role = "用户" if msg["role"] == "user" else "助手"
            content = msg["content"][:150]  # 每条消息最多150字符
            lines.append(f"{role}: {content}")

        context = "\n".join(lines)
        return context[:max_chars]

    def to_list(self) -> List[Dict[str, str]]:
        """返回原始消息列表（用于LangGraph状态传递）"""
        return self.messages.copy()

    def from_list(self, messages: List[Dict[str, str]]):
        """从列表恢复状态（用于LangGraph状态传递）"""
        self.messages = messages.copy() if messages else []


class LongTermMemory:
    """长期记忆：跨会话持久化用户兴趣和研究偏好"""

    def __init__(self, llm=None, memory_file: str = MEMORY_FILE):
        """
        初始化长期记忆
        :param llm: LLM实例，用于提取兴趣关键词。如果为None，使用简单规则提取
        :param memory_file: 记忆持久化文件路径
        """
        self.llm = llm
        self.memory_file = memory_file
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        """从文件加载记忆"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ 记忆文件损坏，创建新的记忆")
        return {
            "interests": [],
            "history_topics": [],
            "last_session": None,
            "session_count": 0
        }

    def _save(self):
        """保存记忆到文件"""
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def extract_interests(self, query: str, report: str) -> List[str]:
        """
        从对话中提取用户兴趣关键词
        :param query: 用户问题
        :param report: 助手回答
        :return: 兴趣关键词列表
        """
        if self.llm:
            # 用LLM智能提取
            prompt = f"""从以下对话中提取用户的学术研究兴趣（3-5个英文关键词）：

用户问题：{query}
助手回答摘要：{report[:300]}

要求：
1. 关键词要具体，如"transformer"、"attention mechanism"
2. 只输出关键词，逗号分隔
3. 不要解释，不要编号

输出示例：natural language processing, pre-trained models, attention mechanism"""

            try:
                response = self.llm.invoke(prompt)
                interests = [k.strip() for k in response.content.split(",") if k.strip()]
                return interests
            except Exception as e:
                print(f"⚠️ LLM提取兴趣失败: {e}，使用规则提取")

        # Fallback：简单规则提取（从query中提取大写专有名词和关键词）
        import re
        words = re.findall(r'[A-Z][a-zA-Z]+', query)
        # 过滤常见停用词
        stopwords = {"What", "How", "Why", "The", "A", "An", "Is", "Are", "It", "This"}
        interests = [w for w in words if w not in stopwords]
        return interests[:5]

    def update(self, query: str, report: str):
        """
        更新长期记忆
        :param query: 用户问题
        :param report: 助手回答
        """
        # 提取新兴趣
        new_interests = self.extract_interests(query, report)

        # 合并去重，保留最近MAX_INTERESTS个
        all_interests = list(dict.fromkeys(self.data["interests"] + new_interests))
        self.data["interests"] = all_interests[-MAX_INTERESTS:]

        # 记录本次话题
        self.data["history_topics"].append({
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "interests": new_interests
        })
        self.data["history_topics"] = self.data["history_topics"][-MAX_HISTORY:]
        self.data["last_session"] = datetime.now().isoformat()
        self.data["session_count"] = self.data.get("session_count", 0) + 1

        self._save()
        return new_interests

    def get_interests(self, n: int = 3) -> List[str]:
        """
        获取最近n个兴趣关键词
        :param n: 返回数量
        :return: 兴趣关键词列表
        """
        return self.data["interests"][-n:] if self.data["interests"] else []

    def get_context_prompt(self) -> str:
        """
        获取长期记忆上下文，用于追加到Prompt
        :return: 记忆上下文字符串，如果没有则返回空串
        """
        interests = self.get_interests(5)
        if not interests:
            return ""
        return f"用户历史兴趣：{', '.join(interests)}。请优先推荐相关领域论文。"

    def clear(self):
        """清空长期记忆（调试用）"""
        self.data = {"interests": [], "history_topics": [], "last_session": None, "session_count": 0}
        self._save()
        print("🧹 长期记忆已清空")

    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        return {
            "total_interests": len(self.data["interests"]),
            "total_topics": len(self.data["history_topics"]),
            "session_count": self.data.get("session_count", 0),
            "last_session": self.data["last_session"]
        }


# ========== 便捷函数（如果不习惯用类）==========

def create_memory_system(llm=None, max_history: int = 10):
    """
    一键创建记忆系统（短期+长期）
    :param llm: LLM实例
    :param max_history: 短期记忆最大轮数
    :return: (ShortTermMemory实例, LongTermMemory实例)
    """
    stm = ShortTermMemory(max_history=max_history)
    ltm = LongTermMemory(llm=llm)
    return stm, ltm


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("🧪 测试记忆系统...")

    # 测试短期记忆
    stm = ShortTermMemory(max_history=6)  # 保留3轮
    stm.add_turn("介绍一下Transformer", "Transformer是一种...")
    stm.add_turn("它和RNN相比呢？", "Transformer相比RNN...")
    print(f"\n短期记忆上下文：\n{stm.get_context()}")

    # 测试长期记忆（无LLM，用规则提取）
    ltm = LongTermMemory(llm=None)
    ltm.update("介绍一下Transformer", "Transformer是一种基于Attention的模型")
    ltm.update("BERT和GPT的区别？", "BERT使用双向Attention，GPT使用因果Attention")

    print(f"\n长期记忆兴趣：{ltm.get_interests()}")
    print(f"记忆统计：{ltm.get_stats()}")
    print(f"Prompt上下文：{ltm.get_context_prompt()}")