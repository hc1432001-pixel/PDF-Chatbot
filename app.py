import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import io
import hashlib
from groq import Groq
from dotenv import load_dotenv

# load_dotenv() reads a local .env file. On Hugging Face Spaces, Secrets
# are injected directly as environment variables, so this line is a no-op
# there and causes no errors — safe to keep for both environments.
load_dotenv()

st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")
st.title("📄 Multi-PDF Chatbot")
st.caption("Upload one or more PDFs, ask questions, get short cited answers")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------------------------------------------------------------------
# Cached resources — loaded once per app instance, not per question
# ---------------------------------------------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_groq_client():
    return Groq(api_key=GROQ_API_KEY)


# ---------------------------------------------------------------------
# PDF processing — cached by file content, so re-uploading an unchanged
# PDF (e.g. after a Streamlit rerun) never re-extracts or re-chunks it
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def process_pdf(file_bytes: bytes, filename: str):
    reader = PdfReader(io.BytesIO(file_bytes))
    chunk_size = 1000
    overlap = 150
    chunks = []

    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if not page_text.strip():
            continue
        start = 0
        while start < len(page_text):
            end = start + chunk_size
            chunks.append({
                "text": page_text[start:end],
                "source": filename,
                "page": page_num + 1,
            })
            start = end - overlap  # overlap prevents cutting answers in half

    return chunks


def file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


# ---------------------------------------------------------------------
# Session state — this is what "remembers" PDFs and chat across reruns
# ---------------------------------------------------------------------
defaults = {
    "all_chunks": [],          # every chunk from every uploaded PDF
    "index": None,             # FAISS index over all_chunks
    "processed_hashes": set(), # which files we've already embedded
    "chat_history": [],        # [{question, answer, sources}]
    "uploaded_names": [],      # display list in sidebar
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

embed_model = load_embedding_model()

# ---------------------------------------------------------------------
# Sidebar — multi-file upload + status + reset
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Your documents")
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        newly_added = False
        for f in uploaded_files:
            f_bytes = f.getvalue()
            h = file_hash(f_bytes)
            if h not in st.session_state.processed_hashes:
                with st.spinner(f"Processing {f.name}..."):
                    chunks = process_pdf(f_bytes, f.name)
                    st.session_state.all_chunks.extend(chunks)
                    st.session_state.processed_hashes.add(h)
                    st.session_state.uploaded_names.append(f.name)
                    newly_added = True

        if newly_added:
            with st.spinner("Building search index..."):
                texts = [c["text"] for c in st.session_state.all_chunks]
                embeddings = embed_model.encode(
                    texts, show_progress_bar=False, batch_size=32
                ).astype("float32")
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(embeddings)
                st.session_state.index = index
            st.success(
                f"{len(st.session_state.uploaded_names)} PDF(s) ready "
                f"· {len(st.session_state.all_chunks)} chunks indexed"
            )

    if st.session_state.uploaded_names:
        st.markdown("**Loaded documents:**")
        for name in st.session_state.uploaded_names:
            st.markdown(f"- {name}")

    if st.button("Clear everything"):
        st.session_state.all_chunks = []
        st.session_state.index = None
        st.session_state.processed_hashes = set()
        st.session_state.chat_history = []
        st.session_state.uploaded_names = []
        st.rerun()

# ---------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------
if not GROQ_API_KEY:
    st.warning(
        "GROQ_API_KEY not found. Add it to a local .env file, or to "
        "Space Settings → Variables and secrets if deployed on Hugging Face."
    )
elif st.session_state.index is None:
    st.info("Upload at least one PDF in the sidebar to start asking questions.")
else:
    # Replay previous turns so the conversation persists across reruns
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn["sources"]:
                with st.expander("Sources"):
                    for s in turn["sources"]:
                        st.markdown(f"**{s['source']} — page {s['page']}**")
                        st.caption(s["text"][:200] + "...")

    question = st.chat_input("Ask a question about your PDFs...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        # --- Retrieve ---
        q_embedding = embed_model.encode([question]).astype("float32")
        k = min(4, len(st.session_state.all_chunks))
        _, indices = st.session_state.index.search(q_embedding, k=k)
        retrieved = [st.session_state.all_chunks[i] for i in indices[0]]

        context = "\n\n".join(
            f"[{r['source']}, page {r['page']}]: {r['text']}" for r in retrieved
        )

        # Short rolling memory — lets the model handle follow-up questions
        # like "what about the second one?" without re-explaining context
        recent_history = ""
        for turn in st.session_state.chat_history[-3:]:
            recent_history += f"Q: {turn['question']}\nA: {turn['answer']}\n"

        prompt = f"""You are answering questions using ONLY the context below,
taken from the user's uploaded PDF(s).
Rules:
- Keep the answer short: 2 to 4 sentences maximum
- Always name which document and page the answer comes from
- If the answer isn't in the context, say so directly — do not guess
Previous conversation:
{recent_history}
Context from PDFs:
{context}
Question: {question}
Concise answer:"""

        # --- Generate ---
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                client = load_groq_client()
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=300,
                )
                answer = response.choices[0].message.content

            st.write(answer)
            with st.expander("Sources"):
                for r in retrieved:
                    st.markdown(f"**{r['source']} — page {r['page']}**")
                    st.caption(r["text"][:200] + "...")

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "sources": retrieved,
        })
