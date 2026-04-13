"""
summarizer.py
Summarizes long transcripts using DistilBART (sshleifer/distilbart-cnn-12-6).
Uses adaptive recursive chunking to handle arbitrarily long text.
"""

from transformers import BartTokenizer, BartForConditionalGeneration

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

# Cached model and tokenizer to avoid reloading on every call
_tokenizer = None
_model = None


def _get_model_and_tokenizer():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        print(f"Loading summarization model: {MODEL_NAME} ...")
        _tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
        _model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    return _tokenizer, _model


def summarize(text: str, max_summary_length: int = 200, min_summary_length: int = 50) -> str:
    """Summarize a single text chunk using DistilBART."""
    tokenizer, model = _get_model_and_tokenizer()

    inputs = tokenizer.encode(
        "summarize: " + text,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    )

    summary_ids = model.generate(
        inputs,
        max_length=max_summary_length,
        min_length=min_summary_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def adaptive_recursive_summarize(
    text: str,
    chunk_size: int = 800,
    min_words: int = 150,
    max_words: int = 400,
) -> str:
    """
    Split long text into chunks, summarize each, then adaptively refine
    the combined result to stay within a reasonable word count range.
    """
    words = text.split()
    chunks = [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

    partial_summaries = [summarize(chunk, max_summary_length=max_words) for chunk in chunks]
    combined = " ".join(partial_summaries)
    word_count = len(combined.split())

    if word_count < min_words:
        # Too short — expand
        return summarize(text, max_summary_length=max_words * 2)
    elif word_count > max_words * 3:
        # Too long — compress
        return summarize(combined, max_summary_length=max_words)
    else:
        return combined


def summarize_url(text: str) -> str:
    """Public entry point: summarize a full transcript string."""
    return adaptive_recursive_summarize(text)
