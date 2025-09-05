import os
import base64
import gc
import tempfile
import uuid
import datetime

from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.core import Settings, VectorStoreIndex, PromptTemplate

import streamlit as st

# ================== AUTH ==================
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = None

if not st.user.is_logged_in and st.session_state.auth_mode != "guest":
    st.warning("Please log in to continue, or try as guest.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Log in"):
            st.login()
    with col2:
        if st.button("Continue as Guest"):
            st.session_state.auth_mode = "guest"
            st.rerun()

    st.stop()
else:
    if st.user.is_logged_in:
        st.session_state.auth_mode = "user"
        user_id = st.user.email or st.user.name
        daily_limit = 5  # logged-in users
        st.sidebar.success(f"Logged in as {st.user.name}")
        if st.button("Log out"):
            st.logout()
            st.rerun()
    elif st.session_state.auth_mode == "guest":
        user_id = "guest"
        daily_limit = 3  # guests get fewer queries
        st.sidebar.warning("You are using Guest mode (3 queries/day).")
        if st.button("End Guest Session"):
            # FIX: also clear guest usage and messages on end
            st.session_state.pop(f"usage_{user_id}", None)
            st.session_state.pop("messages", None)
            st.session_state.auth_mode = None
            st.rerun()

# ================== DAILY LIMIT ==================
today = datetime.date.today()
if f"usage_{user_id}" not in st.session_state:
    st.session_state[f"usage_{user_id}"] = {"count": 0, "date": today}

if st.session_state[f"usage_{user_id}"]["date"] != today:
    st.session_state[f"usage_{user_id}"] = {"count": 0, "date": today}

usage = st.session_state[f"usage_{user_id}"]

if usage["count"] >= daily_limit:
    st.warning("⚠️ You have reached your daily query limit. Please try again tomorrow.")
    st.stop()

st.sidebar.write(f"📊 Queries used today: {usage['count']}/{daily_limit}")  # FIX: daily_limit, not DAILY_LIMIT

# ================== KEYS / SESSION ==================
load_dotenv()
#groq_api_key = st.secrets["GROQ_API_KEY"]
groq_api_key = os.getenv("GROQ_API_KEY")

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
if "file_cache" not in st.session_state:
    st.session_state.file_cache = {}
# FIX: keep query engine in session so chat works after reruns
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

session_id = st.session_state.id

@st.cache_resource
def load_llm():
    return Groq(model="llama-3.3-70b-versatile", api_key=groq_api_key, request_timeout=120.0)

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

def display_pdf_bytes(file_bytes: bytes):
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
    pdf_display = f"""
        <iframe src="data:application/pdf;base64,{base64_pdf}"
                width="100%" height="100%" style="height:100vh; border:none"></iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)

# ================== SIDEBAR: UPLOAD ==================
with st.sidebar:
    st.header("Add your documents:")
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        try:
            file_bytes = uploaded_file.getvalue()
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(file_bytes)

                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")

                if file_key not in st.session_state.file_cache:
                    loader = PyMuPDFReader()
                    docs = loader.load_data(file_path)

                    # Setup llm and embedding model
                    llm = load_llm()
                    embed_model = HuggingFaceEmbedding(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        trust_remote_code=True
                    )
                    Settings.llm = llm
                    Settings.embed_model = embed_model

                    # Build index
                    index = VectorStoreIndex.from_documents(docs, show_progress=True)
                    query_engine = index.as_query_engine(streaming=True, similarity_top_k=4)

                    # Custom prompt
                    qa_prompt_tmpl_str = (
                        "Context information is below.\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "Given the context information above I want you to think step by step to answer the query in a crisp manner, "
                        "incase case you don't know the answer say 'I don't know!'.\n"
                        "Query: {query_str}\n"
                        "Answer: "
                    )
                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
                    query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                    )

                    st.session_state.file_cache[file_key] = query_engine
                    st.session_state.query_engine = query_engine  # FIX: persist engine
                else:
                    st.session_state.query_engine = st.session_state.file_cache[file_key]

                st.success("Ready to chat!")
                display_pdf_bytes(file_bytes)  # FIX: pass bytes, not file object
        except Exception as e:
            st.error(f"An error occured: {e}")
            st.stop()

# ================== MAIN ==================
col1, col2 = st.columns([6, 1])
with col1:
    st.header("Chat with docs")
with col2:
    st.button("Clear", on_click=reset_chat)

if "messages" not in st.session_state:
    reset_chat()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Disable input if no engine yet
qe = st.session_state.query_engine
disabled = qe is None
if disabled:
    st.info("Upload a PDF to start chatting.")

prompt = st.chat_input("What's up?", disabled=disabled)
if prompt:
    # Increment usage BEFORE processing (so the top-of-script check blocks the next request when limit reached)
    usage["count"] += 1
    st.session_state[f"usage_{user_id}"] = usage

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        streaming_response = qe.query(prompt)  # FIX: use persisted qe
        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
