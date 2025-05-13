import fitz  # PyMuPDF
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

genai.configure(api_key="xxxxx")

PDF_PATH='./data/nvdia_10q.pdf'

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    return text.strip()

model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

raw_text = extract_text_from_pdf(PDF_PATH)
cleaned_text = clean_text(raw_text)
chunks = chunk_text(cleaned_text)


def get_embeddings(text_list):
    return model.encode(text_list, convert_to_numpy=True)


def search_faiss(query, top_k=3):
    query_embedding = np.array(get_embeddings([query]), dtype=np.float32)
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

embeddings = get_embeddings(chunks)

# Convert embeddings to numpy array
embeddings_np = np.array(embeddings, dtype=np.float32)

# Initialize FAISS index
index = faiss.IndexFlatL2(embeddings_np.shape[1])
index.add(embeddings_np)


def generate_summary(query):
    relevant_texts = search_faiss(query)
    text_to_summarize = " ".join(relevant_texts)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(text_to_summarize)
    return response.text

query = "Summarize revenue growth and key financial metrics"
summary = generate_summary(query)

print(f"Here is the Summary: {summary}")


query = "Give me insight into Commitments and Contingencies"
summary = generate_summary(query)

print(f"Commitments and Contingencies: {summary}")
