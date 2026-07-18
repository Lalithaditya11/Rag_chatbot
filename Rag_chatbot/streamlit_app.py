import re
from pathlib import Path

import streamlit as st
from chatbot import (
    answer_with_local_model,
    build_context as chatbot_build_context,
    retrieve as chatbot_retrieve,
)

# ====================================================
# CONFIG
# ====================================================

BASE_DIR = Path(__file__).resolve().parent

# ====================================================
# RETRIEVE
# ====================================================

def retrieve(question):
    return chatbot_retrieve(question)


# ====================================================
# CONTEXT
# ====================================================

def build_context(results):
    context, citations = chatbot_build_context(results)
    retrieved = []
    for result in results:
        if isinstance(result[0], dict):
            payload, score = result
        else:
            payload, score = result[0].payload, result[1]
        retrieved.append({
            "score": round(float(score), 2),
            "pdf": payload["filename"],
            "page": payload["page_number"],
            "text": payload["text"][:300],
        })
    return context, citations, retrieved


def generate_answer(question, context, llm_instance):
    return answer_with_local_model(question, context)

# ====================================================
# UI
# ====================================================
st.set_page_config(
    page_title="PDF RAG Chatbot",
    layout="wide"
)
st.title("📚LALLI RAG CHATBOT")
st.write("Ask questions from your PDF knowledge base.")
question = st.text_input(
    "Enter your Question"
)
if st.button("Ask"):
    if question.strip() == "":
        st.warning("Enter a question")
    else:
        with st.spinner("Searching..."):
            results = retrieve(question)
            context, citations, retrieved = build_context(results)
            prompt = f"""
You are a helpful assistant.
Answer ONLY from the provided context.
If the answer is missing, say: I couldn't find this information in the provided PDFs.

Context:
{context}

Question:
{question}
"""
            answer = generate_answer(question, context, llm)
        st.success("Answer")
        st.write(answer)
        st.subheader("Sources")
        for c in sorted(set(citations)):
            st.write("•", c)
        st.subheader("Retrieved Chunks")
        for item in retrieved:
            with st.expander(
                f"{item['pdf']} | Page {item['page']} | Score {item['score']}"
            ):
                st.write(item["text"])