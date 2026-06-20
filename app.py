import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model_gemini = genai.GenerativeModel("gemini-2.5-flash")

# Streamlit UI
st.set_page_config(page_title="PDF Chatbot")
st.title("📄 PDF Chatbot with Gemini")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file:

    # Read PDF
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    st.success("PDF Loaded Successfully")

    # Split text into chunks
    chunk_size = 500
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    # Create embeddings
    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    embeddings = model.encode(chunks)

    # Create FAISS index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        np.array(embeddings).astype("float32")
    )

    # User question
    question = st.text_input(
        "Ask a question about the PDF"
    )

    if question:

        # Convert question to embedding
        query_embedding = model.encode(
            [question]
        )

        # Search similar chunks
        distances, indices = index.search(
            np.array(query_embedding).astype("float32"),
            k=3
        )

        # Build context
        context = "\n\n".join(
            [chunks[idx] for idx in indices[0]]
        )

        # Prompt for Gemini
        prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""

        # Gemini response
        response = model_gemini.generate_content(
            prompt
        )

        st.subheader("Answer")
        st.write(response.text)

        with st.expander("Retrieved Context"):
            st.write(context)
            st.session_state["context"] = context
