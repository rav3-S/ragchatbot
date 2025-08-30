import os
import requests

import random
import base64
import gc
import time
import tempfile
import uuid

from dotenv import load_dotenv
from pydantic import Field
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, PromptTemplate, SimpleDirectoryReader

import streamlit as st
# Load apikey from .env

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None   

@st.cache_resource
def load_llm():
    return Groq(model="llama-3.3-70b-versatile", api_key=groq_api_key ,request_timeout=120.0)


# ==================== Custom Embedding Wrapper ====================
# class OpenTextEmbedding(BaseEmbedding):
#     model: str = Field(default="bge-large-en")
#     api_url: str = Field(default="https://api.opentextembeddings.com/v1")

#     def _get_embedding(self, text):
#         headers = {"Content-Type": "application/json"}
#         payload = {"model": self.model, "input": text}
#         try:
#             r = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
#             if r.status_code == 200:
#                 return r.json()["data"]
#             else:
#                 st.error(f"Embedding API error: {r.status_code}, {r.text}")
#                 return [0.0] * 1024  # fallback vector
#         except Exception as e:
#             st.error(f"Embedding request failed: {e}")
#             return [0.0] * 1024

#     def _get_text_embedding(self, text: str):
#         return self._get_embedding(text)

#     def _get_query_embedding(self, query: str):
#         return self._get_embedding(query)
    
#     async def _aget_query_embedding(self, query: str):
#         return self._call_api(query)


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

                    if(os.path.exists(temp_dir)):
                        # load data
                        loader = SimpleDirectoryReader(
                                    input_dir = temp_dir,
                                    required_exts = [".pdf"],
                                    recursive = True
                                )
                    else:
                        st.error("Could not find the uploaded file. Please try again.")
                        st.stop()

                    docs = loader.load_data()

                    print("DOCS:", docs[0]) 

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