<p align="center">
  <img src="assets/logo-full.png" alt="Legal Compliance AI Agent logo" width="280">
</p>

# 🚀 Legal Compliance AI Agent

> Agentic AI | RAG | Memory | Guardrails | Groq LLM

---

## 📌 Project Overview

Legal Compliance AI Agent is an enterprise AI-powered web application developed as part of the IIT Mandi AI Program Mini Project.

The application enables organizations to upload compliance documents and ask questions using natural language.

The AI retrieves relevant information from uploaded documents, remembers previous conversations, and provides accurate responses with source citations.

---

## ✨ Features

- AI Powered Compliance Assistant
- PDF Upload
- Retrieval Augmented Generation (RAG)
- Conversation Memory
- Guardrails
- Source Citations
- Groq LLM Integration
- Streamlit Web Interface

---

## 🏗 Architecture

```
User
   │
   ▼
Streamlit UI
   │
   ▼
Guardrails
   │
   ▼
Memory
   │
   ▼
Retriever
   │
   ▼
FAISS Vector Database
   │
   ▼
Groq LLM
   │
   ▼
Response
```

---

## ⚙ Tech Stack

| Layer | Technology |
|--------|------------|
| Frontend | Streamlit |
| Backend | Python |
| AI Framework | LangChain |
| LLM | Groq |
| Vector Database | FAISS |
| Embeddings | Sentence Transformers |

---

## 🚀 Installation

Clone Repository

```bash
git clone https://github.com/ankur-gaurav161418/ComplianceAgent-.git
```

Open Folder

```bash
cd ComplianceAgent-
```

Create Virtual Environment

```bash
python -m venv venv
```

Activate

```bash
venv\Scripts\activate
```

Install Packages

```bash
pip install -r requirements.txt
```

Run

```bash
python create_vector_db.py
```

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```text
ComplianceAgent
│
├── app.py
├── create_vector_db.py
├── requirements.txt
├── README.md
├── assets
├── data
├── utils
└── screenshots
```

---

## 🔮 Future Scope

- Multi-Agent AI
- Contract Analyzer
- Risk Assessment
- Compliance Auditor
- Cross-border Compliance Advisor

---

## 👨‍💻 Author

Ankur Gaurav

IIT Mandi AI Program
