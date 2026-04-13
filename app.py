"""
app.py — VidSense AI · Streamlit Interface

Run with:
    streamlit run app.py
"""

import streamlit as st
from Video.downloader import download_audio, compress_audio
from Video.transcriber import speech_to_text
from Video.summarizer import summarize_url
from Video.embeddings import generate_embeddings, closest
from Video.llm import answer_llm

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VidSense AI",
    page_icon="🎬",
    layout="wide",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e2e8f0;
    }

    /* Hero title */
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        text-align: center;
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    /* Cards */
    .glass-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        margin-bottom: 1.5rem;
    }

    /* Chat bubbles */
    .chat-user {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        padding: 0.75rem 1.1rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.4rem 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.95rem;
    }
    .chat-bot {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.1);
        color: #e2e8f0;
        padding: 0.75rem 1.1rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.4rem 0;
        max-width: 80%;
        font-size: 0.95rem;
    }

    /* Input */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        color: white !important;
        border-radius: 10px !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6d28d9, #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .badge-ready { background: #065f46; color: #6ee7b7; }
    .badge-pending { background: #1e3a5f; color: #93c5fd; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Session state defaults ──────────────────────────────────────────────────
for key, default in {
    "transcript": None,
    "summary": None,
    "db": None,
    "chat_history": [],          # list of {"role": "user"|"bot", "text": str}
    "video_url": "",
    "processed": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎬 VidSense AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Drop a YouTube URL · Get a summary · Ask anything about the video</div>',
    unsafe_allow_html=True,
)

# ─── Layout: two columns ─────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.4], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — URL input + summary
# ══════════════════════════════════════════════════════════════════════════════
with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🔗 YouTube Video")

    url_input = st.text_input(
        "Paste a YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        key="url_input_field",
        label_visibility="collapsed",
    )

    process_btn = st.button("⚡ Process Video", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Process pipeline ──────────────────────────────────────────────────────
    if process_btn and url_input:
        st.session_state.video_url = url_input
        st.session_state.processed = False
        st.session_state.chat_history = []

        try:
            with st.status("Processing your video…", expanded=True) as status:
                st.write("📥 Downloading audio…")
                audio = download_audio(url_input)

                st.write("🗜️ Compressing audio…")
                compressed = compress_audio(audio)

                st.write("🎙️ Transcribing (Whisper)…")
                transcript = speech_to_text(compressed)
                st.session_state.transcript = transcript

                st.write("📝 Summarising transcript…")
                summary = summarize_url(transcript)
                st.session_state.summary = summary

                st.write("🔍 Building vector store (ChromaDB)…")
                db = generate_embeddings(transcript)
                st.session_state.db = db

                st.session_state.processed = True
                status.update(label="✅ Ready to chat!", state="complete")

        except Exception as exc:
            st.error(f"❌ Error during processing: {exc}")

    # ── Summary panel ─────────────────────────────────────────────────────────
    if st.session_state.summary:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Video Summary")
        st.markdown(st.session_state.summary)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Transcript expander ───────────────────────────────────────────────────
    if st.session_state.transcript:
        with st.expander("📜 Full Transcript"):
            st.text_area(
                "Transcript",
                value=st.session_state.transcript,
                height=250,
                label_visibility="collapsed",
            )

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — Chat interface
# ══════════════════════════════════════════════════════════════════════════════
with right_col:
    # Status tag
    if st.session_state.processed:
        badge = '<span class="status-badge badge-ready">● Ready</span>'
    else:
        badge = '<span class="status-badge badge-pending">○ Awaiting video</span>'

    st.markdown(f'### 💬 Ask Questions {badge}', unsafe_allow_html=True)

    # Chat history display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">🧑 {msg["text"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-bot">🤖 {msg["text"]}</div>',
                    unsafe_allow_html=True,
                )

    # Input area pinned to bottom of column
    st.markdown("---")
    with st.form(key="chat_form", clear_on_submit=True):
        q_col, btn_col = st.columns([5, 1])
        with q_col:
            user_question = st.text_input(
                "Ask a question",
                placeholder="What does the speaker say about…?",
                label_visibility="collapsed",
                key="question_input",
            )
        with btn_col:
            send_btn = st.form_submit_button("Send")

    if send_btn and user_question:
        if not st.session_state.processed:
            st.warning("⚠️ Please process a video first!")
        else:
            # Store user message
            st.session_state.chat_history.append(
                {"role": "user", "text": user_question}
            )
            with st.spinner("Thinking…"):
                chunks = closest(user_question, st.session_state.db)
                answer = answer_llm(user_question, chunks)
            st.session_state.chat_history.append(
                {"role": "bot", "text": answer}
            )
            st.rerun()

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <hr style="border-color:rgba(255,255,255,0.1);margin-top:3rem"/>
    <div style="text-align:center;color:#475569;font-size:0.8rem;">
        VidSense AI · Powered by Whisper · ChromaDB · Gemini
    </div>
    """,
    unsafe_allow_html=True,
)
