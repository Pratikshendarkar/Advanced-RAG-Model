# 🚀 Advanced RAG Model: An Agentic LLM-Driven Retrieval-Augmented Document Intelligence System

<div align="center">

![Advanced RAG](https://img.shields.io/badge/RAG-Advanced-purple?style=for-the-badge)
![AI Powered](https://img.shields.io/badge/AI-Powered-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/streamlit-1.39.0-red.svg?style=for-the-badge&logo=streamlit)

**An intelligent, production-ready document chat application powered by state-of-the-art Retrieval-Augmented Generation (RAG) techniques**

[Live Demo](#-demo) • [Features](#-features) • [Architecture](#-architecture)

</div>

---

## 🌟 Overview

Advanced RAG PDF Chat is a cutting-edge conversational AI system that transforms how you interact with documents. Built on enterprise-grade RAG architecture, it combines multiple advanced techniques to deliver accurate, contextual answers from your PDF documents with unprecedented precision.

### 🎯 What Makes This Advanced?

Unlike basic PDF chatbots, our system implements **state-of-the-art retrieval techniques** used by leading AI companies:

- **Multi-Stage Retrieval Pipeline** – Combines dense retrieval, query expansion, and re-ranking  
- **Hybrid Search Architecture** – Semantic + keyword-based search for maximum recall  
- **Cross-Encoder Re-ranking** – Precision-focused document scoring  
- **Intelligent Query Optimization** – AI-powered query enhancement and decomposition  
- **Contextual Compression** – Extracts only the most relevant information  
- **Source Attribution** – Full transparency with page-level citations  
- **🆕 Document Transformation** – AI-powered editing, reformatting, summarization, and translation
- **🆕 Presentation Generation** – Auto-create professional PowerPoint presentations from PDFs
- **🆕 Q&A Generation** – Generate comprehensive study materials and quizzes

---

## ✨ Features

### 🔍 **Advanced Retrieval Techniques**

#### 1. **Query Enhancement & Optimization**
- 🧠 AI-powered query expansion  
- 📊 Semantic intent understanding  
- 🎯 Keyword extraction  
- 💡 Context-aware reformulation  

#### 2. **Multi-Query Retrieval**
- 🔄 Parallel query generation  
- 📈 Increased recall  
- 🎪 Diverse semantic coverage  
- 🔀 Reciprocal Rank Fusion (RRF)  

#### 3. **Intelligent Re-Ranking**
- ⚡ Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)  
- 🎯 Precision-based scoring  
- 📊 Confidence ranking  
- 🔝 Top-K selection  

#### 4. **Semantic Search & Embeddings**
- 🌐 Dense vector retrieval  
- 📐 Cosine similarity  
- 🗄️ FAISS Vector DB  
- 🔬 768D embeddings  

---

### 🎨 **Document Transformation Suite** 🆕

Transform your PDFs with AI-powered editing capabilities:

| Type | Description | Use Case |
|------|-------------|----------|
| **📝 Reformat** | Better structure & organization | Messy docs → Clean reports |
| **✍️ Rewrite** | Improve clarity & grammar | Drafts → Professional docs |
| **📊 Summarize** | Condense to key points | 50 pages → 2-page summary |
| **➕ Expand** | Add details & examples | Notes → Full document |
| **🔍 Extract** | Pull specific information | Full doc → Key data |
| **🌍 Translate** | Convert to any language | English → Spanish/French |
| **🎯 Custom** | Your instructions | Resume → Cover letter |

**Features:**
- ✅ Download as PDF or TXT
- ✅ View before/after comparison
- ✅ Professional formatting
- ✅ Preserves important information

---

### 🎨 **Presentation Generation** 🆕

Auto-create professional PowerPoint presentations from your PDFs:

- **📑 5-20 customizable slides**
- **🎭 4 style presets**: Professional, Academic, Creative, Minimalist
- **📝 Smart content extraction**: Text → Bullet points
- **🎤 Speaker notes** for each slide
- **🎨 Professional formatting**: Consistent fonts, colors, spacing
- **💾 Download as .pptx**

**Perfect for:**
- Academic presentations from research papers
- Business decks from reports
- Training materials from documentation
- Lecture slides from textbooks

---

### ❓ **Q&A Generation** 🆕

Generate comprehensive study materials and assessment tools:

- **📝 5-50 customizable Q&A pairs**
- **📊 3 difficulty levels**: Easy, Medium, Hard
- **🎯 3 question types**: Factual, Conceptual, Analytical
- **✅ Complete answers** (2-4 sentences each)
- **📄 Download formats**: PDF and TXT
- **👀 Preview before download**

**Use cases:**
- Students: Create study guides from textbooks
- Teachers: Generate quizzes and exam questions
- Trainers: Build assessment materials
- Compliance: Create knowledge checks from policies

---

## 💬 Conversational Intelligence

- 🧵 Conversation memory  
- 🔄 Follow-up understanding  
- 💭 Intent tracking  
- 📝 Persistent chat history  

---

## 📄 Document Processing

- 📖 Advanced PDF parsing  
- ✂️ Semantic chunking  
- 🔗 Overlap strategy  
- 📏 Optimized chunk size  
- 🧩 Metadata preservation  

---

## 🎨 User Experience

- 💬 ChatGPT-style UI  
- 🎭 Message animations  
- 📱 Responsive layout  
- 🎛️ Advanced controls & toggles  

---

## 🚀 Demo

<img width="1882" height="935" alt="Demo Screenshot" src="https://github.com/user-attachments/assets/a2ece399-2755-4791-a76a-ab4ff4f25ecb" />

---

## 🏗️ Architecture

### 🔁 RAG Pipeline Overview

<p align="center">
  <img src="https://github.com/user-attachments/assets/bbeae8f6-8ba1-44f6-886d-13ac74c99ce9" 
       alt="RAG Pipeline Architecture"
       width="600"/>
</p>

**Pipeline Breakdown:**

1. **Query Optimization Layer**
   - Query enhancement  
   - Multi-query generation  
   - Keyword extraction  

2. **Retrieval Layer**
   - Semantic encoding  
   - FAISS vector search  
   - Similarity matching  

3. **Re-Ranking Layer**
   - Cross-encoder scoring  
   - Relevance sorting  
   - Top-K selection  

4. **Generation Layer**
   - Context formation  
   - LLM (Gemini 2.5 Flash)  
   - Response formatting  

5. **Final Output**
   - Answer with page-level citations  

---

## 🧠 Why This Architecture?

✔ High recall + high precision  
✔ Enterprise-grade scalability  
✔ Transparent source attribution  
✔ Modular & extensible design  

---

## 📦 Installation

### Prerequisites
- Python 3.11+
- Google API Key ([Get one here](https://aistudio.google.com/apikey))

### Setup
```bash
# Clone repository
git clone https://github.com/PratikShendarkar/advanced-rag-pdf-chat.git
cd advanced-rag-pdf-chat

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GOOGLE_API_KEY=your_api_key_here" > .env

# Run application
streamlit run app.py
```

---

## 🛠️ Document Tools Usage

### 1. Transform Document
1. Upload PDFs → Click "Process"
2. Sidebar → Select "📝 Transform Document"
3. Choose type (Reformat/Rewrite/Summarize/Custom)
4. Click "Transform"
5. Download PDF or TXT

### 2. Create Presentation
1. Upload PDFs → Click "Process"
2. Sidebar → Select "🎨 Create Presentation"
3. Set slides (5-20) & style
4. Click "Generate Presentation"
5. Download .pptx

### 3. Generate Q&A
1. Upload PDFs → Click "Process"
2. Sidebar → Select "❓ Generate Q&A"
3. Set number (5-50) & difficulty
4. Click "Generate Q&A"
5. Download PDF or TXT

---

## 🔧 Technology Stack

- **Frontend**: Streamlit 1.39.0
- **LLM**: Google Gemini 2.5 Flash
- **Embeddings**: all-MiniLM-L6-v2 (768-dim)
- **Re-ranker**: ms-marco-MiniLM-L-6-v2
- **Vector DB**: FAISS
- **PDF Processing**: PyPDF2, ReportLab
- **Presentations**: python-pptx



---

⭐ **If you like this project, consider starring the repo!**

</div>
