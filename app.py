import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
from ui import setup_ui, display_chat_messages, render_admin_panel
from utils import (
    load_vectorstore,
    extract_text_from_file,
    whisper_transcribe,
    elevenlabs_tts,
)
from agent import get_agent_response
import database

load_dotenv(override=True)

try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY") or st.secrets.get("ELEVENLABS_API_KEY")
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

LANGUAGES = {
    "🌐 Auto-Detect": "auto",
    "🇺🇸 English": "en",
    "🇪🇸 Spanish": "es",
    "🇫🇷 French": "fr",
    "🇩🇪 German": "de",
    "🇮🇳 Hindi": "hi",
    "🇮🇳 Gujarati": "gu",
    "🇮🇳 Marathi": "mr",
    "🇮🇳 Tamil": "ta",
    "🇮🇳 Telugu": "te",
    "🇮🇳 Bengali": "bn",
    "🇨🇳 Chinese": "zh-cn",
    "🇯🇵 Japanese": "ja",
    "🇦🇪 Arabic": "ar",
}

VECTOR_STORE_DIR = "vectorstore"
VECTOR_STORE_PATH = os.path.join(VECTOR_STORE_DIR, "company_vectorstore")
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Initialize database on startup
database.init_db()

# Session state defaults
for key, default in [
    ("chat_history", []),
    ("conversation_history", []),
    ("vectorstore", None),
    ("uploaded_files", []),
    ("knowledge_updated", False),
    ("files", []),
    ("language_code", "en"),
    ("session_id", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Create or resume a chat session
if st.session_state.session_id is None:
    st.session_state.session_id = database.create_session()
    st.session_state.chat_history = []
    st.session_state.conversation_history = []
else:
    # Load persisted messages on first load
    if not st.session_state.chat_history:
        messages = database.get_session_messages(st.session_state.session_id)
        if messages:
            st.session_state.chat_history = messages
            st.session_state.conversation_history = [
                (msg["content"], nxt["content"])
                for msg, nxt in zip(messages[::2], messages[1::2])
                if msg["role"] == "user" and nxt["role"] == "assistant"
            ]

if st.session_state.vectorstore is None:
    with st.spinner("Initializing knowledge base..."):
        st.session_state.vectorstore = load_vectorstore()


def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "en"


def send_query(query, enable_voice=False):
    if not query.strip():
        return

    # Save user message
    st.session_state.chat_history.append({"role": "user", "content": query})
    database.save_message(
        session_id=st.session_state.session_id, role="user", content=query
    )

    selected_lang = st.session_state.get("selected_language", "🌐 Auto-Detect")
    lang_code = LANGUAGES.get(selected_lang, "auto")
    if lang_code == "auto":
        language_code = detect_language(query)
    else:
        language_code = lang_code

    st.session_state.language_code = language_code

    # Get agent response
    answer, agent_reasoning = get_agent_response(
        query,
        lang=language_code,
        conversation_history=st.session_state.conversation_history,
    )
    audio_file = None
    if enable_voice and answer and ELEVENLABS_API_KEY:
        audio_file = elevenlabs_tts(answer, lang=language_code)

    # Save assistant message
    assistant_msg = {
        "role": "assistant",
        "content": answer,
        "audio": audio_file,
        "agent_reasoning": agent_reasoning,
    }
    st.session_state.chat_history.append(assistant_msg)
    database.save_message(
        session_id=st.session_state.session_id,
        role="assistant",
        content=answer,
        agent_reasoning=agent_reasoning,
        audio_path=audio_file,
    )

    st.session_state.conversation_history.append((query, answer))


def switch_session(session_id):
    st.session_state.session_id = session_id
    messages = database.get_session_messages(session_id)
    st.session_state.chat_history = messages
    st.session_state.conversation_history = [
        (msg["content"], nxt["content"])
        for msg, nxt in zip(messages[::2], messages[1::2])
        if msg["role"] == "user" and nxt["role"] == "assistant"
    ]


def new_chat():
    st.session_state.session_id = database.create_session()
    st.session_state.chat_history = []
    st.session_state.conversation_history = []


def remove_file(filename):
    filepath = os.path.join("uploads", filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    st.session_state.uploaded_files = [
        f for f in st.session_state.uploaded_files if f.get("name") != filename
    ]
    st.session_state.files = [
        f for f in st.session_state.files if f.get("name") != filename
    ]
    st.session_state.knowledge_updated = True
    st.rerun()


def main():
    setup_ui()

    # Admin mode
    is_admin = st.query_params.get("admin") == "true"
    if is_admin:
        admin_pass = os.getenv("ADMIN_PASSWORD", "secret")
        if st.session_state.get("admin_authenticated", False):
            render_admin_panel(None, remove_file, extract_text_from_file)
            st.markdown("---")
        else:
            pwd = st.text_input("Admin Password", type="password")
            if st.button("Login"):
                if pwd == admin_pass:
                    st.session_state["admin_authenticated"] = True
                    st.rerun()
                else:
                    st.error("Invalid password")
            return

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🤖 Customer Assistant")
        st.caption("AI-powered customer support")

        st.divider()

        st.markdown("**🌐 Response Language**")
        st.selectbox(
            "Language",
            options=list(LANGUAGES.keys()),
            key="selected_language",
            label_visibility="collapsed",
        )

        st.divider()

        if st.button("✨ New Chat", type="primary", use_container_width=True):
            new_chat()
            st.rerun()

        st.markdown("**Recent Conversations**")

        sessions = database.list_sessions(limit=15)
        current_sid = st.session_state.session_id

        for sess in sessions:
            title = sess["title"] or "New Chat"
            is_active = sess["session_id"] == current_sid
            icon = "💬" if is_active else "○"

            col_s, col_d = st.columns([5, 1])
            with col_s:
                btn_type = "primary" if is_active else "secondary"
                if st.button(
                    f"{icon} {title[:28]}",
                    type=btn_type,
                    use_container_width=True,
                    key=f"s_{sess['session_id']}",
                ):
                    if not is_active:
                        switch_session(sess["session_id"])
                        st.rerun()
            with col_d:
                if st.button("🗑", key=f"d_{sess['session_id']}"):
                    database.delete_session(sess["session_id"])
                    if sess["session_id"] == current_sid:
                        new_chat()
                    st.rerun()

    # ── Main Area ────────────────────────────────────────────
    st.title("💬 Customer Assistant")
    st.caption("Your AI-powered assistant for Date AI products, orders, and services.")

    # Display chat
    display_chat_messages()

    # Voice input (above chat bar)
    if "voice_transcript" not in st.session_state:
        st.session_state.voice_transcript = ""

    col_voice, _ = st.columns([1, 5])
    with col_voice:
        if st.button("🎤 Voice Input", use_container_width=True):
            with st.spinner("🎙️ Listening..."):
                try:
                    transcript = whisper_transcribe()
                    if transcript and transcript.strip():
                        # Provide immediate feedback
                        st.chat_message("user", avatar="🧑").write(transcript.strip())
                        with st.chat_message("assistant", avatar="🤖"):
                            with st.spinner("Thinking..."):
                                try:
                                    send_query(transcript.strip())
                                except Exception as e:
                                    st.error(f"I encountered an error connecting to my brain: {e}")
                                    st.stop()
                        st.rerun()
                    else:
                        st.warning("No speech detected. Try again.")
                except Exception as e:
                    st.error(f"Voice input error: {e}")

    # Chat input (native Streamlit)
    if prompt := st.chat_input("Ask me anything..."):
        # Provide immediate feedback
        st.chat_message("user", avatar="🧑").write(prompt)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    send_query(prompt)
                except Exception as e:
                    st.error(f"I encountered an error connecting to my brain: {e}")
                    st.stop()
        st.rerun()


if __name__ == "__main__":
    main()