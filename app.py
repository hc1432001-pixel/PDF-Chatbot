import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv

# =========================
# Load Environment Variables
# =========================
load_dotenv()

# =========================
# Gemini Setup
# =========================
genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model_gemini = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# =========================
# Cache Embedding Model
# =========================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

# =========================
# Streamlit UI
# =========================
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄"
)

st.title("📄 PDF Chatbot with Gemini")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

# =========================
# Process PDF Only Once
# =========================
if uploaded_file:

    current_file = uploaded_file.name

    if (
        "processed_file" not in st.session_state
        or st.session_state["processed_file"] != current_file
    ):

        st.info("Processing PDF...")

        # Read PDF
        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        # Chunking
        chunk_size = 1000

        chunks = []

        for i in range(0, len(text), chunk_size):
            chunks.append(
                text[i:i + chunk_size]
            )

        # Load embedding model
        model = load_embedding_model()

        # Create embeddings
        embeddings = model.encode(
            chunks,
            show_progress_bar=False
        )

        embeddings = np.array(
            embeddings
        ).astype("float32")

        # Create FAISS index
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(
            dimension
        )

        index.add(embeddings)

        # Save to session
        st.session_state["processed_file"] = current_file
        st.session_state["chunks"] = chunks
        st.session_state["index"] = index

        st.success(
            f"PDF processed successfully! ({len(chunks)} chunks)"
        )

    else:
        st.success(
            "PDF Loaded Successfully"
        )

    # =========================
    # Question Section
    # =========================
    question = st.text_input(
        "Ask a question about the PDF"
    )

    if question:

        chunks = st.session_state["chunks"]
        index = st.session_state["index"]

        model = load_embedding_model()

        # Convert question to embedding
        query_embedding = model.encode(
            [question]
        )

        query_embedding = np.array(
            query_embedding
        ).astype("float32")

        # Search
        distances, indices = index.search(
            query_embedding,
            k=3
        )

        # Context
        context = "\n\n".join(
            [chunks[idx] for idx in indices[0]]
        )

        # Prompt
        prompt = f"""
You are a PDF assistant.

Answer ONLY from the context below.

Context:
{context}

Question:
{question}

Answer:
"""

        with st.spinner(
            "Generating answer..."
        ):

            response = model_gemini.generate_content(
                prompt
            )

        st.subheader("Answer")

        st.write(
            response.text
        )

        with st.expander(
            "Retrieved Context"
        ):
            st.write(context)