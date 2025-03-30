# URL Q&A Tool

A web-based tool that allows users to ask questions about the content of provided URLs. This tool uses LLMs through Ollama to generate answers based solely on the information from the provided web pages.


## Features

- Input multiple URLs to analyze their content
- Ask questions about the information contained in those URLs
- Get concise answers based only on the ingested information
- View the sources used to generate the answer

## Requirements

- Python 3.8+
- [Ollama](https://ollama.com/download) installed and running
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/url-qa-tool.git
cd url-qa-tool
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Make sure Ollama is installed and running, then pull the required models:
```bash
ollama pull qwen2.5:7b  # or another model of your choice
ollama pull nomic-embed-text
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Enter one or more URLs (one per line) in the URLs text area

4. Type your question about the content of those URLs

5. Click "Get Answer" to receive a response

## How It Works

1. The tool fetches the content from the provided URLs
2. It splits the content into chunks for processing
3. It creates embeddings of these chunks using Ollama
4. When you ask a question, it retrieves the most relevant chunks
5. It sends these chunks along with your question to the LLM
6. The LLM generates an answer based solely on the provided context

## Project Structure

- `app.py`: Main Streamlit application
- `qa_engine.py`: Core functionality for URL processing and question answering
- `requirements.txt`: Python dependencies

## Limitations

- The tool can only answer questions based on the text content of the provided URLs
- Some websites may block web scraping
- The quality of answers depends on the quality of the LLM model used

## License

MIT
