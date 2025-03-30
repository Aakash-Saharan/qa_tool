# app.py
import streamlit as st
import os
import time
from typing import List

# Import the QA engine functionality
from qa_engine import (
    is_ollama_running,
    get_available_models,
    process_urls_and_question,
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBED_MODEL
)

# Set USER_AGENT environment variable
os.environ["USER_AGENT"] = "URL-QA-Tool/1.0"

# Page configuration
st.set_page_config(
    page_title="URL Q&A Tool",
    page_icon="🔍",
    layout="wide"
)

st.title("URL Q&A Tool")
st.markdown("Enter URLs and ask questions about their content")

# Check Ollama status
if not is_ollama_running():
    st.error("⚠️ Ollama service is not running. Please start Ollama and refresh this page.")
    st.info("To install Ollama, visit: https://ollama.com/download")
    st.stop()

# Get available models
available_models = get_available_models()
if not available_models:
    st.warning("⚠️ Could not retrieve models from Ollama. Using default models.")
    available_models = [DEFAULT_LLM_MODEL, "llama3:8b", "mistral:7b"]

# Create sidebar with instructions and settings
with st.sidebar:
    st.header("How to use this tool")
    st.markdown("""
    1. **Enter URLs**: Add one or more URLs (one per line) in the URLs box
    2. **Ask a question**: Type your question about the content of those pages
    3. **Get your answer**: Click "Get Answer" to receive a response based only on the content of the provided URLs
    
    ### Tips for better results
    
    - Be specific with your questions
    - Provide URLs that contain relevant information to your question
    - For complex topics, include multiple URLs with complementary information
    """)
    
    st.header("Model Settings")
    llm_model = st.selectbox(
        "Select Model", 
        options=available_models,
        index=available_models.index(DEFAULT_LLM_MODEL) if DEFAULT_LLM_MODEL in available_models else 0,
        help="Choose the model to use for answering questions"
    )
    
    st.header("Prerequisites")
    st.markdown(f"""
    - Ollama must be installed and running
    - Required models must be pulled:
      ```
      ollama pull {llm_model}
      ollama pull {DEFAULT_EMBED_MODEL}
      ```
    """)
    
    # Show available models
    with st.expander("Available Ollama Models"):
        if available_models:
            for model in available_models:
                st.markdown(f"- {model}")
        else:
            st.markdown("No models found. Make sure Ollama is running.")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    urls_input = st.text_area(
        "URLs (one per line)",
        placeholder="https://example.com\nhttps://another-example.com",
        height=150
    )
    
    question_input = st.text_input(
        "Question",
        placeholder="What is the main topic discussed in these pages?"
    )
    
    submit_button = st.button("Get Answer", type="primary")

# Example section
with st.expander("Examples"):
    example1_col, example2_col = st.columns(2)
    
    with example1_col:
        if st.button("Example 1: AI Information"):
            st.session_state.urls_input = "https://en.wikipedia.org/wiki/Artificial_intelligence"
            st.session_state.question_input = "What is artificial intelligence?"
            st.experimental_rerun()
    
    with example2_col:
        if st.button("Example 2: News Headlines"):
            st.session_state.urls_input = "https://www.cnn.com\nhttps://www.bbc.com"
            st.session_state.question_input = "What are the top news stories today?"
            st.experimental_rerun()

# Set session state for examples
if 'urls_input' in st.session_state:
    urls_input = st.session_state.urls_input
    
if 'question_input' in st.session_state:
    question_input = st.session_state.question_input

# Display results
with col2:
    st.subheader("Answer")
    
    if submit_button:
        # Validate inputs
        if not urls_input.strip():
            st.error("Please enter at least one valid URL.")
        elif not question_input.strip():
            st.error("Please enter a question.")
        else:
            # Parse URLs (one per line)
            urls = [url.strip() for url in urls_input.split('\n') if url.strip().startswith(('http://', 'https://'))]
            
            if not urls:
                st.error("Please enter valid URLs that start with http:// or https://")
            else:
                # Process the query
                start_time = time.time()
                result = process_urls_and_question(
                    urls, 
                    question_input, 
                    llm_model, 
                    DEFAULT_EMBED_MODEL
                )
                end_time = time.time()
                
                if result:
                    st.markdown(f"**Answer:** {result['answer']}")
                    
                    st.subheader("Sources")
                    for source in result['sources']:
                        st.markdown(f"- {source}")
                    
                    st.info(f"Processing time: {end_time - start_time:.2f} seconds")