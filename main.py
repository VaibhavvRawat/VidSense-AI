"""
main.py — VidChat entry point

Usage:
    python main.py --url "https://www.youtube.com/watch?v=..." --question "Your question"

Pipeline:
    1. Download & compress audio from YouTube
    2. Transcribe to text via Whisper
    3. Summarize the transcript
    4. Embed transcript into ChromaDB
    5. Answer questions using Gemini via RAG
"""

import argparse
from Video.downloader import download_audio, compress_audio
from Video.transcriber import speech_to_text
from Video.summarizer import summarize_url
from Video.embeddings import generate_embeddings, closest
from Video.llm import answer_llm


def prepare_transcript(url: str) -> str:
    """Download, compress, and transcribe a YouTube video."""
    print("📥 Downloading audio...")
    audio_file = download_audio(url)

    print("🗜️  Compressing audio...")
    compressed = compress_audio(audio_file)

    print("🎙️  Transcribing audio...")
    transcript = speech_to_text(compressed)

    return transcript


def qna(question: str, db) -> str:
    """Retrieve relevant chunks and answer a question using Gemini."""
    chunks = closest(question, db)
    return answer_llm(question, chunks)


def main():
    parser = argparse.ArgumentParser(description="VidChat — Chat with any YouTube video")
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument("--question", default=None, help="Question to ask about the video")
    parser.add_argument("--summarize", action="store_true", help="Print a summary of the video")
    args = parser.parse_args()

    transcript = prepare_transcript(args.url)

    if args.summarize:
        print("\n📝 Summary:\n")
        print(summarize_url(transcript))

    print("\n🔍 Building vector store...")
    db = generate_embeddings(transcript)

    if args.question:
        print(f"\n❓ Question: {args.question}\n")
        print("💬 Answer:\n")
        print(qna(args.question, db))
    else:
        # Interactive Q&A loop
        print("\n💬 Interactive Q&A mode (type 'exit' to quit)\n")
        while True:
            question = input("You: ").strip()
            if question.lower() in ("exit", "quit", "q"):
                break
            if question:
                print("Bot:", qna(question, db), "\n")


if __name__ == "__main__":
    main()
