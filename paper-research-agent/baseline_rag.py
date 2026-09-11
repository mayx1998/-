# baseline_rag.py
import os
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ========== API配置 ==========
API_KEY = os.getenv("DASHSCOPE_API_KEY")
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

print(f"✅ API Key: {API_KEY[:15]}...")

embeddings = DashScopeEmbeddings(model="text-embedding-v3", dashscope_api_key=API_KEY)
llm = ChatOpenAI(model="qwen-plus", api_key=API_KEY, base_url=API_BASE, temperature=0.7)

# ========== 1. 加载论文 ==========
print("\n📚 加载论文...")

paper1 = Document(
    page_content="""We propose the Transformer, based solely on attention mechanisms. 
    An attention function maps a query and key-value pairs to an output. 
    The output is a weighted sum of values, where weights come from a compatibility function 
    of the query with the corresponding key. We use Scaled Dot-Product Attention: 
    compute dot products of query with all keys, divide by sqrt(d_k), apply softmax.""",
    metadata={"Title": "Attention Is All You Need", "Year": "2017"}
)

paper2 = Document(
    page_content="""BERT uses bidirectional attention and two pre-training tasks: 
    Masked Language Model (MLM) and Next Sentence Prediction (NSP). 
    Unlike GPT which uses left-to-right causal attention, BERT can see both left and right context.""",
    metadata={"Title": "BERT: Pre-training of Deep Bidirectional Transformers", "Year": "2019"}
)

paper3 = Document(
    page_content="""GPT uses a Transformer decoder with causal (autoregressive) masking. 
    The pre-training objective is next token prediction. This left-to-right attention limits 
    the model's ability to use future context, but makes it naturally suited for generation tasks.""",
    metadata={"Title": "Improving Language Understanding by Generative Pre-Training", "Year": "2018"}
)

docs = [paper1, paper2, paper3]
print(f"✅ 加载了 {len(docs)} 篇论文")

# ========== 2. 切分 ==========
print("\n✂️  切分文本...")
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"✅ 切分为 {len(chunks)} 段")

# ========== 3. 生成向量（手动实现内存向量存储）==========
print("\n🧠 生成向量...")

# 获取所有文本
chunk_texts = [chunk.page_content for chunk in chunks]
# 批量生成向量
chunk_embeddings = embeddings.embed_documents(chunk_texts)
print(f"✅ 生成了 {len(chunk_embeddings)} 个向量，维度: {len(chunk_embeddings[0])}")

# 手动实现余弦相似度检索
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query, top_k=3):
    query_vec = embeddings.embed_query(query)
    scores = [(i, cosine_similarity(query_vec, chunk_emb))
              for i, chunk_emb in enumerate(chunk_embeddings)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [chunks[i] for i, _ in scores[:top_k]]

# ========== 4. RAG链 ==========
def format_docs(docs_list):
    return "\n\n".join([f"[{i+1}] {d.metadata['Title']}: {d.page_content[:300]}"
                        for i, d in enumerate(docs_list)])

prompt = ChatPromptTemplate.from_template("""基于以下论文片段回答问题：

{context}

问题：{question}
要求：1.用中文回答 2.标注来源[1][2]... 3.信息不足时明确说明

回答：""")

# 自定义检索函数
def custom_retrieve(query):
    return retrieve(query, top_k=3)

chain = (
    {
        "context": lambda q: format_docs(custom_retrieve(q)),
        "question": RunnablePassthrough()
    }
    | prompt | llm | StrOutputParser()
)

# ========== 5. 提问 ==========
print("\n❓ 提问: Transformer中的Attention机制原理是什么？")
print("="*60)
result = chain.invoke("Transformer中的Attention机制原理是什么？")
print(result)
print("="*60)

print("\n✅ 基线RAG跑通！下一步：升级为Agentic RAG")

