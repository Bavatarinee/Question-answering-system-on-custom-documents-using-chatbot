# Question-answering-system-on-custom-documents-using-chatbot

Here is a **simple, clean, human-style GitHub README** for your project — no AI-style wording, no fancy marketing phrases, just a straightforward technical description.

---

# **Document Question Answering System (NLP + ML)**

This project is a simple Question Answering system that lets users upload a PDF or text file and ask questions based on the content. The application extracts text from the document, processes it using basic NLP techniques, and finds the most relevant answer using TF-IDF and cosine similarity.
A Streamlit interface is used to make the system easy to use.

---

## **Features**

* Upload PDF or TXT files
* Automatic text extraction
* Sentence-based chunking with overlap
* TF-IDF vectorization of document chunks
* Cosine similarity to find the closest answer
* Clean UI built with Streamlit
* Answers displayed in bullet format

---

## **Tech Stack**

* **Python**
* **Streamlit**
* **NLTK**
* **Scikit-learn**
* **PyPDF2**
* **Regular Expressions**

---

## **How It Works**

1. The user uploads a document.
2. Text is extracted (PDF or TXT).
3. Text is cleaned and split into sentences.
4. Sentences are grouped into overlapping chunks.
5. TF-IDF transforms each chunk into vector form.
6. When the user enters a question:

   * The question is vectorized
   * Cosine similarity is calculated
   * The most relevant chunks are selected
7. Sentences from those chunks are displayed as the answer.

---

## **Installation**



### **1. Install dependencies**

```bash
pip install -r requirements.txt
```

### **2. Download NLTK data**

```python
import nltk
nltk.download("punkt")
```

---

## **Run the Application**

```bash
streamlit run qa_app.py
```

---

## **Project Structure**

```
├── qa_app.py
├── requirements.txt
└── README.md
```

---

## **Usage**

1. Run the Streamlit app.
2. Upload a PDF or text document.
3. Type your question in the input box.
4. The system displays the most relevant answer from the document.

---

## **Example Output**

* Extracted key sentences shown as bullet points
* Highest similarity score shown for reference

---

## **Notes**

* Works best on text-heavy documents.
* Very long PDFs may take a few seconds to process.
* This is a retrieval-based QA system, not a generative model.

---
