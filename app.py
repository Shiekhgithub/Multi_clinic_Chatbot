"""
app.py
------
Streamlit chatbot UI for the Healthcare RAG system.

Run:
    streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from vector_store import load_all_stores, stores_exist
from agent import build_agent

load_dotenv()

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare RAG Chatbot",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Healthcare RAG Chatbot")
st.caption(
    "Powered by LangChain · ChromaDB · sentence-transformers  |  "
    "Datasets: Heart Disease · Dermatology · Pakistani Diabetes"
)

# ──────────────────────────────────────────────
# Sidebar — configuration
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    provider = st.selectbox(
    "LLM Provider",
    ["groq", "openai", "gemini"],
    index=["groq", "openai", "gemini"].index(
        os.getenv("LLM_PROVIDER", "groq")
    ) if os.getenv("LLM_PROVIDER", "groq") in ["groq", "openai", "gemini"] else 0,
)

    k_docs = st.slider("Documents to retrieve (k)", min_value=2, max_value=10, value=5)

    show_trace = st.checkbox("Show reasoning trace", value=False)

    st.divider()
    st.markdown("**Datasets loaded:**")
    st.markdown("- 🫀 Heart Disease")
    st.markdown("- 🔬 Dermatology")
    st.markdown("- 🩸 Pakistani Diabetes")

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.agent_executor = None
        st.rerun()

# ──────────────────────────────────────────────
# Initialise session state
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None

if "stores_loaded" not in st.session_state:
    st.session_state.stores_loaded = False

# ──────────────────────────────────────────────
# Load stores + agent (once per session)
# ──────────────────────────────────────────────
persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

if not st.session_state.stores_loaded:
    if not stores_exist(persist_dir):
        st.error(
            "⚠️ Vector stores not found. "
            "Please run `python ingest.py` first to build the ChromaDB collections."
        )
        st.stop()

    with st.spinner("Loading vector stores and initialising agent …"):
        stores = load_all_stores(persist_dir)
        st.session_state.agent_executor = build_agent(stores, provider=provider, k=k_docs)
        st.session_state.stores_loaded = True

# ──────────────────────────────────────────────
# Display chat history
# ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ──────────────────────────────────────────────
# Chat input
# ──────────────────────────────────────────────
example_questions = [
    "What are common symptoms in patients with heart disease?",
    "Tell me about patients with psoriasis.",
    "What glucose levels are associated with diabetic patients?",
    "Compare cholesterol levels between heart disease and non-heart-disease patients.",
]

with st.expander("💡 Example questions"):
    for q in example_questions:
        if st.button(q, key=q):
            st.session_state._prefill = q

user_input = st.chat_input("Ask about heart disease, skin conditions, or diabetes …")

# Handle prefill from example buttons
if hasattr(st.session_state, "_prefill") and st.session_state._prefill:
    user_input = st.session_state._prefill
    del st.session_state._prefill

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            try:
                result = st.session_state.agent_executor.invoke({"messages": [{"role": "user", "content": user_input}]},{"configurable": {"thread_id": "1"}})
                # answer = result
                messages = result.get("messages", [])

                if messages:
                    last_message = messages[-1]

                    if isinstance(last_message.content, list):
                        answer = last_message.content[0].get("text", "")
                    else:
                        answer = last_message.content
                else:
                    answer = "No response generated."

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # Optionally show intermediate steps
                if show_trace and "intermediate_steps" in result:
                    with st.expander("🔍 Reasoning trace"):
                        for step in result["intermediate_steps"]:
                            action, obs = step
                            st.markdown(f"**Tool:** `{action.tool}`")
                            st.markdown(f"**Query:** {action.tool_input}")
                            st.markdown(f"**Retrieved:** {str(obs)[:500]} …")
                            st.divider()

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                err_msg = f"⚠️ Error: {e}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
