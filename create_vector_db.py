"""
create_vector_db.py
-------------------
Steps:
  1. Read all PDF files from the "data/" folder
  2. Split each document into smaller text chunks
  3. Generate vector embeddings with sentence-transformers
  4. Build a FAISS vector database from those embeddings
  5. Save the index and metadata to the "vectorstore/" folder
Install dependencies:
  pip install pymupdf sentence-transformers faiss-cpu numpy
"""
import os
import json
import numpy as np
import fitz
from sentence_transformers import SentenceTransformer
import faiss
DATA_FOLDER = "data"
VECTORSTORE_FOLDER = "vectorstore"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
def load_pdfs(data_folder):
    if not os.path.exists(data_folder):
        raise FileNotFoundError(
            "Folder '{}' not found. Create it and add PDF files.".format(data_folder)
        )
    pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise ValueError("No PDF files found in '{}'.".format(data_folder))
    print("Found {} PDF(s) in '{}':".format(len(pdf_files), data_folder))
    documents = []
    for filename in sorted(pdf_files):
        filepath = os.path.join(data_folder, filename)
        print("  Reading:", filepath)
        doc = fitz.open(filepath)
        pages = [doc[i].get_text() for i in range(len(doc))]
        doc.close()
        full_text = "\n".join(pages).strip()
        if full_text:
            documents.append({"source": filepath, "text": full_text})
            print("    -> {} chars, {} pages".format(len(full_text), len(pages)))
        else:
            print("    -> WARNING: no text found (may be a scanned image PDF)")
    print("\nDocuments loaded:", len(documents))
    return documents
def chunk_text(text, chunk_size, chunk_overlap):
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks
def chunk_documents(documents, chunk_size, chunk_overlap):
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["text"], chunk_size, chunk_overlap)
        for i, c in enumerate(chunks):
            all_chunks.append({"source": doc["source"], "chunk_id": i, "text": c})
        print("  '{}' -> {} chunks".format(doc["source"], len(chunks)))
    print("\nTotal chunks:", len(all_chunks))
    return all_chunks
def generate_embeddings(chunks, model_name):
    print("\nLoading model '{}'...".format(model_name))
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    print("Encoding {} chunks...".format(len(texts)))
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    print("Embedding shape:", embeddings.shape)
    return embeddings
def build_faiss_index(embeddings):
    emb = embeddings.astype(np.float32)
    dim = emb.shape[1]
    print("\nBuilding FAISS index (dim={})...".format(dim))
    index = faiss.IndexFlatL2(dim)
    index.add(emb)
    print("Vectors in index:", index.ntotal)
    return index
def save_vectorstore(index, chunks, vectorstore_folder):
    os.makedirs(vectorstore_folder, exist_ok=True)
    index_path = os.path.join(vectorstore_folder, "index.faiss")
    faiss.write_index(index, index_path)
    print("\nSaved FAISS index ->", index_path)
    chunks_path = os.path.join(vectorstore_folder, "chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print("Saved chunk metadata ->", chunks_path)
    print("\nDone! Vector database is ready in '{}/'"
          .format(vectorstore_folder))
def main():
    print("=" * 60)
    print("PDF -> FAISS Vector Database Builder")
    print("Data folder    :", DATA_FOLDER)
    print("Vectorstore    :", VECTORSTORE_FOLDER)
    print("Embedding model:", EMBEDDING_MODEL)
    print("Chunk size     :", CHUNK_SIZE, "chars")
    print("Chunk overlap  :", CHUNK_OVERLAP, "chars")
    print("=" * 60)
    print("\n[1/5] Loading PDFs...")
    documents = load_pdfs(DATA_FOLDER)
    print("\n[2/5] Chunking documents...")
    chunks = chunk_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    print("\n[3/5] Generating embeddings...")
    embeddings = generate_embeddings(chunks, EMBEDDING_MODEL)
    print("\n[4/5] Building FAISS index...")
    index = build_faiss_index(embeddings)
    print("\n[5/5] Saving vector database...")
    save_vectorstore(index, chunks, VECTORSTORE_FOLDER)
if __name__ == "__main__":
    main()