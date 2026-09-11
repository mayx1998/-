# 智能论文检索系统 — All-in-One 镜像（API服务 + 内置前端页面）
FROM python:3.12-slim

WORKDIR /app

# 先拷贝依赖清单，利用Docker层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝源码与评测集
COPY *.py ./
COPY eval_dataset.json ./

EXPOSE 8000

# 通过环境变量传入: -e DASHSCOPE_API_KEY=你的key
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
