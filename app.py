import streamlit as st
from dotenv import load_dotenv, find_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from htmlTemplates import css, bot_template, user_template
import os
from pathlib import Path
from google import genai
from sentence_transformers import CrossEncoder
import time

def get_pdf_text_with_metadata(pdf_docs):
    """Extract text with metadata (page numbers, source)"""
    documents = []
    for pdf_idx, pdf in enumerate(pdf_docs):
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text.strip():
                doc = Document(
                    page_content=text,
                    metadata={
                        'source': pdf.name,
                        'page': page_num + 1,
                        'pdf_index': pdf_idx
                    }
                )
                documents.append(doc)
    return documents

def get_pdf_info(pdf_file):
    """Get basic PDF information"""
    try:
        pdf_file.seek(0)
        pdf_reader = PdfReader(pdf_file)
        return {
            'pages': len(pdf_reader.pages),
            'name': pdf_file.name,
            'size': f"{pdf_file.size / 1024:.1f} KB"
        }
    except Exception as e:
        return None

def get_text_chunks_with_metadata(documents):
    """Split documents into chunks while preserving metadata"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = []
    for doc in documents:
        splits = text_splitter.split_text(doc.page_content)
        for split in splits:
            chunk = Document(
                page_content=split,
                metadata=doc.metadata
            )
            chunks.append(chunk)
    return chunks

def get_vectorstore(chunks):
    """Create vector store with metadata"""
    from langchain_huggingface import HuggingFaceEmbeddings
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if hf_token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def format_docs(docs):
    """Format documents for display"""
    return "\n\n".join([f"[Source: {doc.metadata.get('source', 'Unknown')}, Page {doc.metadata.get('page', '?')}]\n{doc.page_content}" for doc in docs])

def enhance_query(original_query):
    """Improve user's question for better retrieval - ENHANCED PROMPT"""
    try:
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        
        # ENHANCED PROMPT for better query improvement
        prompt = f"""You are an expert at improving search queries to find the most relevant information in documents.

Task: Transform the user's question into an optimized search query that will retrieve the best matching content.

Original question: "{original_query}"

Instructions:
1. Identify the core intent and key concepts
2. Expand abbreviations and add relevant synonyms
3. Make the query more specific and detailed
4. Keep it as a clear, natural language question
5. Focus on searchable keywords

Enhanced question (only output the improved question, nothing else):"""
        
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        if response and response.text:
            enhanced = response.text.strip()
            # Extract important keywords
            keywords = [word for word in original_query.split() if len(word) > 4][:5]
            return enhanced, keywords
    except Exception as e:
        pass
    return original_query, []

def generate_multi_queries(question):
    """Generate multiple query variations - ENHANCED PROMPT"""
    try:
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        
        # ENHANCED PROMPT for better query variations
        prompt = f"""You are an expert at generating diverse search queries to maximize information retrieval.

Original question: "{question}"

Task: Generate 2 alternative versions of this question that:
1. Use different wording but maintain the same intent
2. Approach the topic from different angles
3. Include synonyms and related terms
4. Are specific and clear

Output only the 2 alternative questions, one per line, without numbering or labels:"""
        
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        if response and response.text:
            queries = [question]
            for line in response.text.split('\n'):
                line = line.strip()
                if line and len(line) > 10:
                    cleaned = line
                    for prefix in ['Alternative 1:', 'Alternative 2:', '1.', '2.', '1)', '2)', 'Alternative:', '-']:
                        cleaned = cleaned.replace(prefix, '').strip()
                    if cleaned and cleaned not in queries and len(cleaned) > 10:
                        queries.append(cleaned)
            return queries[:3]
    except Exception as e:
        pass
    return [question]

def rerank_documents(query, documents, top_k=3):
    """Re-rank documents using cross-encoder"""
    if not documents:
        return documents
    try:
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        pairs = [[query, doc.page_content] for doc in documents]
        scores = model.predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked[:top_k]]
    except Exception as e:
        return documents[:top_k]

def query_gemini(context, question, max_retries=2):
    """Query Gemini with ENHANCED PROMPT for better answers"""
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    models_to_try = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-flash-latest']
    
    if 'working_gemini_model' in st.session_state and st.session_state.working_gemini_model:
        models_to_try.insert(0, st.session_state.working_gemini_model)
    
    for model_name in models_to_try:
        for attempt in range(max_retries):
            try:
                max_context = 4000
                if len(context) > max_context:
                    context = context[:max_context] + "\n\n[Context truncated...]"
                
                # ENHANCED PROMPT - Much more detailed instructions for better answers
                prompt = f"""You are an intelligent assistant specialized in analyzing documents and providing accurate, helpful answers.

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Read the context carefully and identify all relevant information
2. Answer the question directly and accurately based ONLY on the provided context
3. If the context contains the answer:
   - Provide a clear, comprehensive response
   - Include specific details, numbers, dates, or facts from the context
   - Structure your answer logically (use paragraphs if needed)
   - Be concise but complete
4. If the context does NOT contain enough information:
   - Clearly state: "Based on the provided documents, I cannot find specific information about [topic]."
   - If there's partial information, share what you found and explain what's missing
5. If asked to compare, list, or analyze:
   - Organize information clearly
   - Use bullet points only if it improves clarity
   - Provide complete explanations
6. Cite specific details from the context when possible (e.g., "According to page X...")
7. Be professional, accurate, and helpful

ANSWER:"""
                
                response = client.models.generate_content(model=model_name, contents=prompt)
                
                if response and hasattr(response, 'text') and response.text:
                    if 'working_gemini_model' not in st.session_state:
                        st.session_state.working_gemini_model = model_name
                    return response.text
                    
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                if "404" in error_msg or "not found" in error_msg.lower():
                    break
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
    
    return "I'm having trouble generating a response. Please try again."

def process_query_advanced(user_question, vectorstore, use_enhancement=True, use_multi_query=True, use_reranking=True):
    """Process query with advanced techniques"""
    results = {
        'original_query': user_question,
        'enhanced_query': user_question,
        'keywords': [],
        'all_docs': [],
        'final_docs': []
    }
    
    if use_enhancement:
        try:
            enhanced, keywords = enhance_query(user_question)
            results['enhanced_query'] = enhanced
            results['keywords'] = keywords
        except:
            pass
    
    queries_to_search = [user_question]
    if use_multi_query:
        try:
            multi_queries = generate_multi_queries(user_question)
            queries_to_search = multi_queries
        except:
            pass
    
    all_docs = []
    seen_content = set()
    for query in queries_to_search:
        try:
            docs = vectorstore.similarity_search(query, k=5)
            for doc in docs:
                if doc.page_content not in seen_content:
                    all_docs.append(doc)
                    seen_content.add(doc.page_content)
        except:
            continue
    
    results['all_docs'] = all_docs
    
    if use_reranking and all_docs:
        try:
            final_docs = rerank_documents(user_question, all_docs, top_k=3)
            results['final_docs'] = final_docs
        except:
            results['final_docs'] = all_docs[:3]
    else:
        results['final_docs'] = all_docs[:3]
    
    return results

def display_chat_messages():
    """Display all chat messages"""
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                st.markdown(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.markdown(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    env_path = find_dotenv()
    if env_path:
        load_dotenv(env_path, override=True)
    else:
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=True)
    
    st.set_page_config(page_title="Advanced RAG Chat", page_icon="💬", layout="wide")
    st.markdown(css, unsafe_allow_html=True)
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("⚠️ GOOGLE_API_KEY not found in .env file!")
        st.stop()
    
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "working_gemini_model" not in st.session_state:
        st.session_state.working_gemini_model = None
    if "pdf_docs" not in st.session_state:
        st.session_state.pdf_docs = None
    if "use_enhancement" not in st.session_state:
        st.session_state.use_enhancement = True
    if "use_multi_query" not in st.session_state:
        st.session_state.use_multi_query = True
    if "use_reranking" not in st.session_state:
        st.session_state.use_reranking = True
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = None
    
    with st.sidebar:
        st.title("⚙️ Settings")
        
        with st.expander("🎛️ Advanced Features", expanded=False):
            st.session_state.use_enhancement = st.checkbox("🔍 Query Enhancement", value=st.session_state.use_enhancement, help="AI improves your question for better search")
            st.session_state.use_multi_query = st.checkbox("🔄 Multi-Query", value=st.session_state.use_multi_query, help="Searches with multiple question variations")
            st.session_state.use_reranking = st.checkbox("⚡ Re-ranking", value=st.session_state.use_reranking, help="Re-orders results by relevance")
        
        st.divider()
        
        st.subheader("📄 Upload Documents")
        pdf_docs = st.file_uploader("Choose PDFs", accept_multiple_files=True, type=['pdf'], label_visibility="collapsed")
        
        if pdf_docs:
            with st.expander("📑 Document Info", expanded=False):
                total_pages = 0
                for pdf in pdf_docs:
                    info = get_pdf_info(pdf)
                    if info:
                        total_pages += info['pages']
                        st.text(f"📄 {info['name'][:30]}... ({info['pages']} pages)")
                st.info(f"Total: {len(pdf_docs)} docs, {total_pages} pages")
        
        if st.button("🚀 Process", type="primary", use_container_width=True):
            if not pdf_docs:
                st.error("Upload PDFs first!")
            else:
                with st.spinner("Processing..."):
                    try:
                        documents = get_pdf_text_with_metadata(pdf_docs)
                        chunks = get_text_chunks_with_metadata(documents)
                        vectorstore = get_vectorstore(chunks)
                        st.session_state.vectorstore = vectorstore
                        st.session_state.pdf_docs = pdf_docs
                        st.session_state.chat_history = []
                        st.success("✅ Ready to chat!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        if st.session_state.chat_history:
            st.divider()
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.last_sources = None
                st.rerun()
        
        if st.session_state.last_sources:
            st.divider()
            with st.expander("📚 Last Sources", expanded=False):
                for i, doc in enumerate(st.session_state.last_sources, 1):
                    st.caption(f"**{i}.** {doc.metadata.get('source', 'Unknown')} - Page {doc.metadata.get('page', '?')}")
    
    st.title("💬 Chat with Your PDFs")
    
    display_chat_messages()
    
    if not st.session_state.chat_history:
        st.info("👋 Upload PDFs in the sidebar and start chatting!")
    
    user_question = st.chat_input("Ask about your documents...")
    
    if user_question:
        if st.session_state.vectorstore is None:
            st.warning("⚠️ Please upload and process PDFs first!")
        else:
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            st.markdown(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
            
            try:
                with st.spinner("🤔 Thinking..."):
                    use_enhancement = st.session_state.use_enhancement
                    use_multi_query = st.session_state.use_multi_query
                    use_reranking = st.session_state.use_reranking
                    
                    results = process_query_advanced(
                        user_question,
                        st.session_state.vectorstore,
                        use_enhancement,
                        use_multi_query,
                        use_reranking
                    )
                    
                    if not results['final_docs']:
                        response = "I couldn't find relevant information in the documents to answer your question."
                    else:
                        context = format_docs(results['final_docs'])
                        response = query_gemini(context, user_question)
                        st.session_state.last_sources = results['final_docs']
                    
                    st.session_state.chat_history.append(AIMessage(content=response))
                    st.markdown(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)
                    st.rerun()
                    
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.session_state.chat_history.append(AIMessage(content=error_msg))
                st.error(error_msg)
                st.rerun()

if __name__ == '__main__':
    main()