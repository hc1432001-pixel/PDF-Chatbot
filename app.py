import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import os
import time

# ==========================
# Gemini Setup
# ==========================
genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model_gemini = genai.GenerativeModel(
    "gemini-1.5-flash"
)

# ==========================
# Streamlit UI
# ==========================
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄"
)

st.title("📄 PDF Chatbot")

# ==========================
# Load Embedding Model Once
# ==========================
@st.cache_resource
def load_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

# ==========================
# Upload PDF
# ==========================
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file:

    # Process PDF only once
    if (
        "pdf_name" not in st.session_state
        or st.session_state["pdf_name"] != uploaded_file.name
    ):

        with st.spinner("Processing PDF..."):

            reader = PdfReader(uploaded_file)

            text = ""

            for page in reader.pages:
                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

            # Chunking
            chunk_size = 1000
            chunks = []

            for i in range(
                0,
                len(text),
                chunk_size
            ):
                chunks.append(
                    text[i:i + chunk_size]
                )

            # Embeddings
            model = load_model()

            embeddings = model.encode(
                chunks,
                show_progress_bar=False
            )

            embeddings = np.array(
                embeddings
            ).astype("float32")

            # FAISS
            index = faiss.IndexFlatL2(
                embeddings.shape[1]
            )

            index.add(embeddings)

            st.session_state["pdf_name"] = uploaded_file.name
            st.session_state["chunks"] = chunks
            st.session_state["index"] = index

        st.success(
            f"PDF Processed Successfully! ({len(chunks)} chunks)"
        )

    # ==========================
    # Ask Question
    # ==========================
    question = st.text_input(
        "Ask a question about the PDF"
    )

    if question:

        model = load_model()

        query_embedding = model.encode(
            [question]
        )

        query_embedding = np.array(
            query_embedding
        ).astype("float32")

        distances, indices = (
            st.session_state["index"].search(
                query_embedding,
                k=3
            )
        )

        context = "\n\n".join(
            [
                st.session_state["chunks"][i]
                for i in indices[0]
            ]
        )

        prompt = f"""
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

            start_time = time.time()

            response = model_gemini.generate_content(
                prompt
            )

            total_time = (
                time.time() - start_time
            )

        st.write(
            f"Response Time: {total_time:.2f} sec"
        )

        st.subheader("Answer")

        st.write(
            response.text
        )

        with st.expander(
            "Retrieved Context"
        ):
            st.write(context)