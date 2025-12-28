# 🚀 Advanced RAG PDF Chat - Enterprise-Grade Document Intelligence System

<div align="center">

![Advanced RAG](https://img.shields.io/badge/RAG-Advanced-purple?style=for-the-badge)
![AI Powered](https://img.shields.io/badge/AI-Powered-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/streamlit-1.39.0-red.svg?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/license-MIT-green.svg?style=for-the-badge)

**An intelligent, production-ready document chat application powered by state-of-the-art Retrieval-Augmented Generation (RAG) techniques**

[Live Demo](#-demo) • [Features](#-features) • [Architecture](#-architecture)

</div>

---

## 🌟 Overview

Advanced RAG PDF Chat is a cutting-edge conversational AI system that transforms how you interact with documents. Built on enterprise-grade RAG architecture, it combines multiple advanced techniques to deliver accurate, contextual answers from your PDF documents with unprecedented precision.

### 🎯 What Makes This Advanced?

Unlike basic PDF chatbots, our system implements **state-of-the-art retrieval techniques** used by leading AI companies:

- **Multi-Stage Retrieval Pipeline** - Combines dense retrieval, query expansion, and re-ranking
- **Hybrid Search Architecture** - Semantic + keyword-based search for maximum recall
- **Cross-Encoder Re-ranking** - Precision-focused document scoring
- **Intelligent Query Optimization** - AI-powered query enhancement and decomposition
- **Contextual Compression** - Extracts only the most relevant information
- **Source Attribution** - Full transparency with page-level citations

---

## ✨ Features

### 🔍 **Advanced Retrieval Techniques**

#### 1. **Query Enhancement & Optimization**
- 🧠 **AI-Powered Query Expansion**: Automatically improves questions by adding synonyms, expanding abbreviations, and clarifying intent
- 📊 **Semantic Understanding**: Analyzes query intent to optimize search strategy
- 🎯 **Keyword Extraction**: Identifies and prioritizes important search terms
- 💡 **Context-Aware Reformulation**: Adapts queries based on document type

#### 2. **Multi-Query Retrieval**
- 🔄 **Parallel Search Variants**: Generates 3-5 alternative phrasings of your question
- 📈 **Increased Recall**: Captures documents that match different query formulations
- 🎪 **Diverse Perspectives**: Approaches questions from multiple angles
- 🔀 **Query Fusion**: Combines results using reciprocal rank fusion (RRF)

#### 3. **Intelligent Re-Ranking**
- ⚡ **Cross-Encoder Scoring**: Uses `ms-marco-MiniLM-L-6-v2` for precise relevance scoring
- 🎯 **Precision Optimization**: Re-orders retrieved chunks by actual relevance
- 📊 **Confidence Scoring**: Assigns relevance scores to each result
- 🔝 **Top-K Selection**: Returns only the most relevant context

#### 4. **Semantic Search & Embeddings**
- 🌐 **Dense Vector Retrieval**: Powered by `sentence-transformers/all-MiniLM-L6-v2`
- 📐 **Cosine Similarity**: Finds semantically similar content
- 🗄️ **FAISS Vector Database**: Lightning-fast similarity search
- 🔬 **768-Dimensional Embeddings**: Rich semantic representation

### 💬 **Conversational Intelligence**

#### 5. **Context-Aware Chat**
- 🧵 **Conversation Memory**: Maintains context across multiple questions
- 🔄 **Follow-up Understanding**: Handles pronouns and references to previous messages
- 💭 **Intent Tracking**: Understands evolving conversation topics
- 📝 **Chat History**: Persistent conversation tracking

#### 6. **Smart Answer Generation**
- 🎨 **Structured Responses**: Organizes information logically
- 📚 **Source Citations**: References specific pages and documents
- ⚠️ **Uncertainty Handling**: Clearly states when information isn't available
- 🎯 **Direct & Concise**: Answers exactly what was asked

### 📄 **Document Processing**

#### 7. **Advanced PDF Parsing**
- 📖 **Multi-Page Support**: Processes documents of any length
- 🔍 **Metadata Extraction**: Captures page numbers, sources, and structure
- 📊 **Table & List Handling**: Preserves document formatting
- 🔤 **Text Normalization**: Cleans and standardizes extracted text

#### 8. **Intelligent Chunking**
- ✂️ **Semantic Splitting**: Breaks documents at natural boundaries
- 🔗 **Overlap Strategy**: 200-character overlap prevents context loss
- 📏 **Optimal Chunk Size**: 1000 characters for balanced context/precision
- 🧩 **Metadata Preservation**: Tracks source and page for each chunk

### 🎨 **User Experience**

#### 9. **ChatGPT-Style Interface**
- 💬 **Modern Chat UI**: Familiar, intuitive design
- 🎭 **Message Animations**: Smooth fade-in effects
- 🎨 **Gradient Bubbles**: User messages in purple, AI in gray
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile

#### 10. **Real-Time Feedback**
- ⏳ **Progress Indicators**: Shows processing stages
- 🔄 **Live Updates**: Real-time message streaming
- 📊 **Processing Stats**: Displays chunks created, pages processed
- 💡 **Status Messages**: Clear feedback at every step

#### 11. **Advanced Controls**
- 🎛️ **Feature Toggles**: Enable/disable query enhancement, multi-query, re-ranking
- 📑 **Source Inspector**: View exact text chunks used for answers
- 🗑️ **Chat Management**: Clear history, start fresh conversations
- 📈 **Statistics Dashboard**: Track usage metrics



## 🚀 Demo

<img width="1882" height="935" alt="image" src="https://github.com/user-attachments/assets/a2ece399-2755-4791-a76a-ab4ff4f25ecb" />


### Sample Conversation
```
User: What are the key findings in the research paper?

Bot: Based on the research paper, the key findings are:

1. **Performance Improvement**: The proposed model achieved 
   15% higher accuracy compared to baseline methods (Page 7)

2. **Efficiency Gains**: Processing time reduced by 40% 
   through optimized architecture (Page 12)

3. **Scalability**: Successfully tested on datasets up to 
   1M samples without degradation (Page 15)

These findings are detailed in the Results section starting 
from page 7.

Sources Used: research_paper.pdf - Pages 7, 12, 15
