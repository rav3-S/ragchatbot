import os
import requests

import random
import base64
import gc
import time
import tempfile
import uuid
import datetime

from dotenv import load_dotenv
from pydantic import Field
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, PromptTemplate, SimpleDirectoryReader

import streamlit as st
import streamlit_authenticator as stauth

# ================== AUTH ==================
if not st.user.is_logged_in:
    st.warning("Please log in to continue")
    if st.button("Log in"):
        st.login()
else:
    st.sidebar.success(f"Logged in as {st.user.name}")
    if st.button("Log out"):
        st.logout()
# ================== END AUTH ==================

# ================== DAILY LIMIT ==================
DAILY_LIMIT = 5
today = datetime.date.today()

# Use st.user.name (or st.user.email) as unique identifier
if st.user.is_logged_in:
    user_id = st.user.email or st.user.name  # fallback to name if email not available

    if f"usage_{user_id}" not in st.session_state:
        st.session_state[f"usage_{user_id}"] = {"count": 0, "date": today}

    # Reset count if a new day
    if st.session_state[f"usage_{user_id}"]["date"] != today:
        st.session_state[f"usage_{user_id}"] = {"count": 0, "date": today}

    # Check usage before allowing chat
    if st.session_state[f"usage_{user_id}"]["count"] >= DAILY_LIMIT:
        st.warning("⚠️ You have reached your daily query limit. Please try again tomorrow.")
        st.stop()


# Load apikey from .env

load_dotenv()

groq_api_key = st.secrets["GROQ_API_KEY"]

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None   

@st.cache_resource
def load_llm():
    return Groq(model="llama-3.3-70b-versatile", api_key=groq_api_key ,request_timeout=120.0)


def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


def display_pdf(file):
    # Opening file from file path

    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)    


with st.sidebar:
    st.header(f"Add your documents:")

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

                if file_key not in st.session_state.get('file_cache', {}):
                    # Load PDF text using PyMuPDFReader
                    loader = PyMuPDFReader()
                    docs = loader.load_data(file_path)  # file_path is the uploaded PDF

                    print("DOCS:", docs[0].text[:500])  # first 500 chars for debug

                    # Setup llm and embedding model
                    llm = load_llm()
                    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", trust_remote_code=True)

                    Settings.llm = llm
                    Settings.embed_model = embed_model

                    # Build index
                    index = VectorStoreIndex.from_documents(docs, show_progress=True)
                    query_engine = index.as_query_engine(streaming=True, similarity_top_k=4)

                    # ====== Customise prompt template ======
                    qa_prompt_tmpl_str = (
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                    )
                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

                    query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                    )

                    st.session_state.file_cache[file_key] = query_engine

                else:
                    query_engine = st.session_state.file_cache[file_key]
                
                # File is now processed and display the pdf
                st.success("Ready to chat!")
                display_pdf(uploaded_file)
        except Exception as e:
            st.error(f"An error occured: {e}")
            st.stop()


col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Chat with docs")

with col2:
    st.button("Clear", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's up?"):
    # check daily quota
    usage = st.session_state[f"usage_{user_id}"]

    if usage["count"] >= DAILY_LIMIT:
        st.warning("⚠️ Daily query limit reached. Come back tomorrow.")
        st.stop()
    
    usage["count"]+=1
    st.session_state[f"usage_{user_id}"] = usage
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Simulate stream of response with milliseconds delay
        streaming_response = query_engine.query(prompt)
        
        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        # full_response = query_engine.query(prompt)

        message_placeholder.markdown(full_response)
        # st.session_state.context = ctx

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})