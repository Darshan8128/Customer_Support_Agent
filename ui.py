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