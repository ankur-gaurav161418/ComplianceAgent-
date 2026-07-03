"""
app.py
------
Streamlit chat app that answers questions from your PDF documents.

How it works:
  1. Loads the FAISS vector database built by create_vector_db.py
  2. When you ask a question, finds the most relevant text chunks
  3. Sends those chunks + conversation history to Ollama (llama3.1:8b)
  4. Streams the answer back and shows which PDFs it came from

Requirements:
  pip install streamlit sentence-transformers faiss-cpu numpy
  # Ollama must be running locally: https://ollama.com
  # Pull the model first: ollama pull llama3.1:8b

Run:
  streamlit run app.py
"""

import os
import json
import requests
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

VECTORSTORE_FOLDER = "vectorstore"
INDEX_FILE = os.path.join(VECTORSTORE_FOLDER, "index.faiss")
CHUNKS_FILE = os.path.join(VECTORSTORE_FOLDER, "chunks.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.1:8b"
MAX_HISTORY = 10


@st.cache_resource
def load_vectorstore():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE):
        return None, None
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def retrieve_chunks(query, index, chunks, model, top_k):
    query_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_vec, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "score": float(dist),
        })
    return results


def build_prompt_messages(history, context_text, user_question):
    system_message = {
        "role": "system",
        "content": (
            "You are a helpful assistant that answers questions based on the"
            " provided document excerpts. Use the excerpts below as your primary"
            " source of information. If the answer is not in the excerpts, say"
            " so clearly.\n\nDOCUMENT EXCERPTS:\n" + context_text
        ),
    }
    recent_history = history[-(MAX_HISTORY * 2):]
    return [system_message] + recent_history + [{"role": "user", "content": user_question}]


def stream_ollama(messages):
    payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": True}
    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        yield "ERROR: Cannot connect to Ollama. Run 'ollama serve' and pull the model."
        return
    except requests.exceptions.RequestException as e:
        yield "ERROR: {}".format(e)
        return
    for line in response.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line)
            token = data.get("message", {}).get("content", "")
            if token:
                yield token
            if data.get("done", False):
                break
        except json.JSONDecodeError:
            continue


def sidebar(chunks):
    with st.sidebar:
        st.title("Document Library")
        st.caption("PDFs indexed in the vector database")
        if chunks is None:
            st.warning("No vector database found. Run create_vector_db.py first.")
            return
        sources = sorted(set(c["source"] for c in chunks))
        if not sources:
            st.info("No documents found in the database.")
        else:
            for src in sources:
                count = sum(1 for c in chunks if c["source"] == src)
                st.markdown("**{}**".format(os.path.basename(src)))
                st.caption("{} chunks indexed".format(count))
        st.divider()
        st.markdown("**Settings**")
        st.caption("Model: {}".format(OLLAMA_MODEL))
        st.caption("Embedding: {}".format(EMBEDDING_MODEL))
        st.caption("Chunks retrieved: {}".format(TOP_K))
        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def main():
    st.set_page_config(page_title="LEGAL Chat Assistant", page_icon="books", layout="wide")
    index, chunks = load_vectorstore()
    embedding_model = load_embedding_model()
    sidebar(chunks)
    st.title("LEGAL Chat Assistant")
    st.caption("Ask questions about your documents. Powered by Ollama + FAISS.")
    if index is None:
        st.warning("Vector database not found. Run create_vector_db.py first.", icon="warning")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources used"):
                    for src in msg["sources"]:
                        st.caption("**{}** -- chunk {} (score {:.4f})".format(
                            os.path.basename(src["source"]), src["chunk_id"], src["score"]))
                        st.text(src["text"][:300] + ("..." if len(src["text"]) > 300 else ""))
    user_input = st.chat_input("Ask a question about your documents...", disabled=(index is None))
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        retrieved = retrieve_chunks(user_input, index, chunks, embedding_model, TOP_K)
        context_text = ""
        for i, r in enumerate(retrieved):
            context_text += "[{}] {} (chunk {}):\n{}\n\n".format(
                i + 1, os.path.basename(r["source"]), r["chunk_id"], r["text"])
        history_for_ollama = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]
        messages = build_prompt_messages(history_for_ollama, context_text, user_input)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            for token in stream_ollama(messages):
                full_response += token
                placeholder.markdown(full_response + "...")
            placeholder.markdown(full_response)
            if retrieved:
                with st.expander("Sources used"):
                    for src in retrieved:
                        st.caption("**{}** -- chunk {} (score {:.4f})".format(
                            os.path.basename(src["source"]), src["chunk_id"], src["score"]))
                        st.text(src["text"][:300] + ("..." if len(src["text"]) > 300 else ""))
        st.session_state.messages.append({
            "role": "assistant", "content": full_response, "sources": retrieved})


if __name__ == "__main__":
    main()

    BLOCKED_TERMS = ["hack", "fraud", "tax evasion", "steal"]
BLOCKED_RESPONSE = "Sorry, I cannot assist with illegal activities."

def check_guardrails(text):
    lowered = text.lower()
    for term in BLOCKED_TERMS:
        if " " in term:
            if term in lowered:          # phrase match
                return True, term
        else:
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, lowered):  # whole-word match
                return True, term
    return False, None
