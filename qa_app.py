import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import nltk
import PyPDF2
import re

nltk.download("punkt")

st.markdown(
    """
    <style>
    .answer-box {
        background-color: #f0fff0;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #53a653;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💬 Improved Document QA (NLP + ML)")
st.write("Upload a text or PDF document and ask questions. The bot will answer based on the document content with enhanced formatting.")

uploaded_file = st.file_uploader("Upload a text or PDF file", type=["txt", "pdf"])

text = ""
if uploaded_file:
    file_type = uploaded_file.type
    if file_type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    elif file_type == "text/plain":
        text = uploaded_file.read().decode("utf-8")
    else:
        st.error("Unsupported file type!")
        st.stop()

    # Simple preprocessing: lower case, remove some punctuations for best tokenization
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.]', '', text)

    if not text.strip():
        st.error("No text could be extracted from the document.")
        st.stop()

    st.write("📄 Document loaded:")
    st.write(text[:500] + "..." if len(text) > 500 else text)

    # Improved chunking: slightly larger, overlapping, paragraph-aware if possible
    sentences = sent_tokenize(text)
    chunk_size = 5  # Larger chunk contains more context
    overlap = 2     # More overlap, so relevant context is preserved
    chunks = [
        " ".join(sentences[i:i+chunk_size])
        for i in range(0, len(sentences)-chunk_size+1, chunk_size-overlap)
    ]
    # Fallback if document is very short
    if not chunks:
        chunks = [text]

    # Advanced tfidf options: wider ngrams, more features
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 3),    # Capture bigrams/trigrams
        max_features=7000      # Increase if your doc is large
    )
    X = vectorizer.fit_transform(chunks)

    question = st.text_input("Ask a question:")

    if st.button("Get Answer"):
        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            # Preprocess the question
            question_pp = question.lower()
            question_pp = re.sub(r'\s+', ' ', question_pp)
            question_pp = re.sub(r'[^\w\s.]', '', question_pp)

            q_vec = vectorizer.transform([question_pp])
            sims = cosine_similarity(q_vec, X)[0]

            top_idxs = sims.argsort()[::-1][:2]
            top_score = sims[top_idxs[0]]
            answer_chunks = [chunks[i] for i in top_idxs]

            # Extract sentences, remove duplicates, and order logically
            seen = set()
            answer_points = []
            for chunk in answer_chunks:
                for line in sent_tokenize(chunk):
                    line = line.strip()
                    if line and line not in seen:
                        answer_points.append(line)
                        seen.add(line)

            # Custom colored answer box with HTML and markdown
            bullet_html = "".join([f"<li>{point}</li>" for point in answer_points])
            st.markdown(
                f"<div class='answer-box'><ul>{bullet_html}</ul></div>",
                unsafe_allow_html=True
            )
            st.info(f"Top similarity score: {top_score:.2f}", icon="✅")
            st.success("Answer provided in logical bullet format.")
