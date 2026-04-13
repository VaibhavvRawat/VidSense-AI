# VidSense AI

VidSense AI is an AI-powered application that allows you to summarize YouTube videos and ask questions about their content. It leverages state-of-the-art models for speech-to-text, summarization, and question answering, making it a powerful tool for extracting insights from video content.

## Features
- **Summarize YouTube Videos:** Automatically transcribe and summarize the content of any YouTube video.
- **Ask Questions:** Query the video content to get answers about specific topics discussed.
- **Embeddings & Search:** Uses embeddings to enable semantic search and context-aware Q&A.

## How It Works
1. **Download Audio:** Extracts audio from a YouTube video using `yt-dlp`.
2. **Transcribe Audio:** Converts audio to text using OpenAI Whisper.
3. **Summarize Content:** Summarizes the transcript using a transformer-based model (BART).
4. **Generate Embeddings:** Creates embeddings for semantic search and Q&A using HuggingFace models and LangChain.
5. **Question Answering:** Uses Gemini LLM to answer questions based on the video content.

## Requirements
- Python 3.8+
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [openai-whisper](https://github.com/openai/whisper)
- [ffmpeg](https://ffmpeg.org/)
- [langchain](https://github.com/hwchase17/langchain)
- [transformers](https://github.com/huggingface/transformers)
- [sentence-transformers](https://www.sbert.net/)
- [google-generativeai](https://github.com/google/generative-ai-python)

## Setup
1. Clone the repository.
2. Install the required Python packages:
	```bash
	pip install -r requirements.txt
	```
3. Copy `.env.example` to `.env` and add your Gemini API key:
	```bash
	cp .env.example .env
	# Then edit .env and set GEMINI_API_KEY=your_key_here
	```

## Usage
```bash
# Summarize a video
python main.py --url "https://www.youtube.com/watch?v=..." --summarize

# Ask a single question
python main.py --url "https://www.youtube.com/watch?v=..." --question "What is discussed?"

# Interactive Q&A mode
python main.py --url "https://www.youtube.com/watch?v=..."
```

## Project Structure
```
VidSense AI/
├── main.py               # Entry point — runs the full pipeline
├── requirements.txt      # All Python dependencies
├── .env.example          # Environment variable template
├── .gitignore
└── Video/                # Core pipeline package
    ├── downloader.py     # YouTube audio download & compression
    ├── transcriber.py    # Whisper speech-to-text
    ├── embeddings.py     # ChromaDB vector store & RAG retrieval
    ├── summarizer.py     # BART-based adaptive summarization
    └── llm.py            # Gemini LLM question answering
```

