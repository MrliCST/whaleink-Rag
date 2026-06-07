# 🧠 whaleink RAG

> **博客 AI 问答系统** — 基于 RAG（检索增强生成）技术，让访问者直接向你的技术博客提问。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5%2B-FF6B35)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ 功能

| 功能 | 说明 |
|------|------|
| 📥 **知识库构建** | 自动扫描博客 Markdown 文章，分割为文档块 |
| 🔍 **智能检索** | TF-IDF + jieba 中文分词 + ChromaDB 向量库 |
| 🤖 **AI 生成** | 基于 DeepSeek API 生成上下文相关回答 |
| 💬 **前端浮窗** | 一键嵌入博客，用户点击 💬 即可提问 |
| 🚀 **轻量部署** | 单机可跑（1.6GB 内存可用），无需 GPU |

## 🏗️ 架构

```
用户浏览器 ──→ Nginx (https://whaleink.top/rag/)
                  │
                  ▼
            FastAPI (port 8000)
                  │
          ┌───────┴───────┐
          ▼               ▼
     ChromaDB          DeepSeek API
    (向量存储)          (LLM 生成)
          │
          ▼
     jieba 分词
   + TF-IDF 检索
```

## 🚀 快速开始

### 1. 克隆并安装

```bash
git clone git@github.com:MrliCST/whaleink-rag.git
cd whaleink-rag
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# DeepSeek API Key（必填）
echo "DEEPSEEK_API_KEY=sk-your-key" >> .env

# ChromaDB 存储路径（可选）
echo "CHROMA_DB_PATH=/path/to/chroma_db" >> .env
```

### 3. 构建知识库

```bash
# 指定博客文章目录
export BLOG_SOURCE_DIR=/path/to/your/blog/source/_posts
python ingest.py
```

### 4. 启动服务

```bash
python app.py
# 访问 http://localhost:8000
```

### 5. 嵌入博客

在页面上添加以下代码，即可出现 💬 问答按钮：

```html
<button class="rag-btn" onclick="toggleRag()">💬</button>
<div class="rag-panel">...（详见 static/index.html）</div>
```

## 📦 项目结构

```
whaleink-rag/
├── app.py               # FastAPI 服务入口
├── rag_pipeline.py      # RAG 核心：Embedding + 检索 + 生成
├── ingest.py            # 知识库构建脚本
├── requirements.txt     # Python 依赖
├── static/              # 前端静态文件
│   └── index.html       # 独立前端页面
├── chroma_db/           # 向量数据库（运行时生成）
└── .env                 # 环境配置
```

## 🔌 API 接口

### `GET /api/status`

服务状态检查。

### `POST /api/query`

```json
{
  "question": "MyBatis 怎么配驼峰命名映射？"
}
```

**响应：**

```json
{
  "answer": "在 application.yml 中配置 ...",
  "sources": ["MyBatis 命名规范与映射规则详解"]
}
```

### `POST /api/reload`

重新加载知识库。

## 🛠️ 技术栈

- **框架：** FastAPI + Uvicorn
- **向量库：** ChromaDB
- **检索：** jieba 分词 + Scikit-learn TfidfVectorizer
- **LLM：** DeepSeek API（OpenAI 兼容）
- **嵌入：** 服务器使用 TF-IDF（免 GPU），也可切换至 HuggingFace API

## 📄 License

MIT
