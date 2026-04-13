"""
llm.py
Interfaces with Google Gemini to answer questions using RAG-retrieved context.
Reads the API key from the GEMINI_API_KEY environment variable.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv
from .embeddings import create_prompt

load_dotenv()

_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    raise EnvironmentError(
        "GEMINI_API_KEY is not set. "
        "Create a .env file with GEMINI_API_KEY=your_key_here"
    )

genai.configure(api_key=_api_key)

GEMINI_MODEL = "gemini-2.5-pro"


def answer_llm(question: str, closest_chunks: list) -> str:
    """
    Generate an answer for `question` grounded in `closest_chunks`
    using the Gemini model.
    """
    prompt = create_prompt(closest_chunks, question)
    if prompt.startswith("Sorry"):
        return prompt

    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)

    if response.candidates and response.candidates[0].content.parts:
        return response.candidates[0].content.parts[0].text

    return "No answer could be generated."
