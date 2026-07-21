import streamlit as st
import os


def setup_ui():
    st.set_page_config(
        page_title="Customer Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject minimal CSS overrides
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown("<style>" + f.read() + "</style>", unsafe_allow_html=True)

    # Inject JS to move the audio_input record button into the chat bar
    st.markdown("""
    <script>
    (function moveMicIntoBar() {
        function tryMove() {
            // Find the record button inside stAudioInput
            const audioInput = document.querySelector('[data-testid="stAudioInput"]');
            const chatInputRow = document.querySelector('[data-testid="stChatInputContainer"] > div');
            const sendBtn = document.querySelector('[data-testid="stChatInputSubmitButton"]');

            if (audioInput && sendBtn && chatInputRow) {
                const recordBtn = audioInput.querySelector('button');
                if (recordBtn && !document.getElementById('__mic_btn_injected__')) {
                    recordBtn.id = '__mic_btn_injected__';
                    recordBtn.title = 'Voice input';
                    // Style the moved button
                    Object.assign(recordBtn.style, {
                        background: 'transparent',
                        border: 'none',
                        color: 'rgba(255,255,255,0.6)',
                        cursor: 'pointer',
                        width: '36px',
                        height: '36px',
                        borderRadius: '50%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        padding: '4px',
                        marginRight: '4px',
                        flexShrink: '0',
                    });
                    // Insert before the send button
                    sendBtn.parentNode.insertBefore(recordBtn, sendBtn);
                    // Hide the original container
                    audioInput.style.display = 'none';
                }
            } else {
                // DOM not ready yet, retry
                setTimeout(tryMove, 200);
            }
        }
        // Run after page load and on any Streamlit reruns
        if (document.readyState === 'complete') { tryMove(); }
        else { window.addEventListener('load', tryMove); }
        // Also observe DOM mutations for Streamlit reruns
        new MutationObserver(tryMove).observe(document.body, {childList: true, subtree: true});
    })();
    </script>
    """, unsafe_allow_html=True)


def render_admin_panel(reset_chat, remove_file, extract_text_from_file):
    pass


def display_chat_messages():
    """Renders chat messages using Streamlit's native st.chat_message."""
    show_reasoning = os.getenv("ADA_DEBUG") == "1"

    if not st.session_state.chat_history:
        # Welcome state
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                "### 💬 Welcome to Customer Assistant\n"
                "I'm here to help you with Date AI products, orders, and support.\n\n"
                "**Try asking:**\n"
                "- 📦 *Check order status (e.g. ORD-1001)*\n"
                "- 💡 *What are Date AI's features?*\n"
                "- 🔧 *Troubleshooting help*\n"
                "- 💰 *Pricing information*"
            )
        st.markdown("---")
        return

    for message in st.session_state.chat_history:
        role = message["role"]
        avatar = "🧑" if role == "user" else "🤖"

        with st.chat_message(role, avatar=avatar):
            st.markdown(message["content"])

            if role == "assistant":
                if "audio" in message and message["audio"]:
                    st.audio(message["audio"])

                if (
                    show_reasoning
                    and "agent_reasoning" in message
                    and message["agent_reasoning"]
                ):
                    reasoning = message["agent_reasoning"]
                    tools_used = reasoning.get("tools_used", [])
                    latency = reasoning.get("total_latency_seconds", "?")
                    tool_names = [
                        t["name"].replace("_tool", "") for t in tools_used
                    ]
                    label = (
                        f"🔍 Agent reasoning — "
                        f"{', '.join(tool_names) if tool_names else 'direct'} "
                        f"({latency}s)"
                    )
                    with st.expander(label):
                        for i, tool_info in enumerate(tools_used):
                            st.markdown(f"**{i+1}. Tool:** `{tool_info['name']}`")
                            args_str = ", ".join(
                                f"{k}=`{str(v)[:80]}`"
                                for k, v in tool_info.get("args", {}).items()
                            )
                            if args_str:
                                st.markdown(f"   **Args:** {args_str}")
                            st.markdown(
                                f"   **Result:** {tool_info.get('result_summary', 'N/A')}"
                            )