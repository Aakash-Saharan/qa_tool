# qa_engine.py
import streamlit as st
import requests
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Default model configuration
DEFAULT_LLM_MODEL = "qwen2.5:7b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"

# Check if Ollama is running
def is_ollama_running() -> bool:
    """Check if Ollama service is running on localhost."""
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=2)
        return response.status_code == 200
    except:
        return False

# Function to get available models from Ollama
def get_available_models() -> List[str]:
    """Get a list of available models from Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model.get("name") for model in models if model.get("name")]
        return []
    except:
        return []

# Cache URL content fetching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_and_process_urls(urls: List[str]):
    """
    Fetch and process URL content, returning documents and splits.
    
    Args:
        urls: List of URLs to fetch and process
        
    Returns:
        tuple: (documents, splits) or (None, None) if processing fails
    """
    try:
        loader = WebBaseLoader(urls)
        loader.requests_kwargs = {'verify': True, 'timeout': 10}
        documents = loader.load()
        
        if not documents:
            return None, None
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        splits = text_splitter.split_documents(documents)
        return documents, splits
    except Exception as e:
        st.error(f"Error loading URLs: {str(e)}")
        return None, None

def process_urls_and_question(
    urls: List[str], 
    question: str, 
    llm_model: str = DEFAULT_LLM_MODEL,
    embedding_model: str = DEFAULT_EMBED_MODEL
) -> Optional[Dict[str, Any]]:
    """
    Process URLs and generate an answer to the question.
    
    Args:
        urls: List of URLs to analyze
        question: The question to answer
        llm_model: Name of the Ollama model to use for answering
        embedding_model: Name of the Ollama model to use for embeddings
        
    Returns:
        dict: Dictionary containing the answer and sources, or None if processing fails
    """
    try:
        with st.spinner("Loading and processing content from URLs..."):
            # Fetch and process URLs
            documents, splits = fetch_and_process_urls(urls)
            
            if not documents or not splits:
                st.error("Could not extract text from any of the provided URLs")
                return None
        
        with st.spinner("Creating embeddings and retrieving information..."):
            # Create vector store with OllamaEmbeddings
            try:
                embeddings = OllamaEmbeddings(model=embedding_model)
                vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            except Exception as e:
                st.error(f"Error creating embeddings: {str(e)}")
                return None
            
            # Create retriever
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            
            # Create QA chain with Ollama
            try:
                llm = Ollama(model=llm_model, temperature=0.3)
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True
                )
            except Exception as e:
                st.error(f"Error creating language model: {str(e)}")
                return None
            
            # Get answer
            result = qa_chain({"query": question})
            
            # Extract sources
            used_sources = list(set([doc.metadata.get("source", "") for doc in result["source_documents"]]))
            
            return {
                "answer": result["result"],
                "sources": used_sources
            }
    
    except Exception as e:
        st.error(f"Error processing request: {str(e)}")
        return None