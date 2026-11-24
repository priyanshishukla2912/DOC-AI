# DocuChat — AI-Powered Document Q&A System

DocuChat is a privacy-first conversational document analysis platform that implements a Retrieval-Augmented Generation (RAG) pipeline with client-side cryptographic protection.

## Key Features

- **Document Formats**: Upload PDF, DOCX, TXT, or Markdown documents.
- **Privacy & Encryption Vault**: AES-256 symmetric encryption at rest. Raw plaintext is never persisted to disk; documents are decrypted strictly in volatile memory during vector indexing.
- **Hybrid RAG Pipeline**: Combines recursive semantic text chunking with Google Gemini neural embeddings and local TF-IDF cosine similarity for resilient retrieval.
- **Multi-Turn Conversational Memory**: Natural multi-turn dialogue with strict anti-hallucination context grounding.
- **Pinpoint Source Attribution**: Cites specific document names, page/section numbers, and relevance percentages for every factual assertion.
- **Automated Summaries**: Automatic executive brief, key themes, and suggested questions upon document ingestion.
- **Zero-Retention Mode**: Secure multi-pass file shredding and in-memory key wiping.

## Architecture

```
User Document (PDF/DOCX/TXT)
          │
          ▼
AES-256 Encryption Vault (.vault/*.enc)
          │
          ▼
Volatile In-Memory Decryption
          │
          ▼
Recursive Semantic Splitter
          │
          ▼
Hybrid Vector Index (Gemini / TF-IDF)
          │
          ▼
Cosine Similarity Top-K Retrieval
          │
          ▼
Anti-Hallucination Prompt Template
          │
          ▼
Google Gemini LLM (Streaming Output)
```

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/priyanshishukla2912/DOC-AI.git
   cd DOC-AI
   ```

2. **Set up a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure your Gemini API key (optional for local search):**
   ```bash
   cp .env.example .env
   # Add your key to .env: GEMINI_API_KEY=your_key_here
   ```

4. **Launch the application:**
   ```bash
   streamlit run app.py
   ```

## Running Tests

```bash
python -m unittest discover tests
```

---
MADE WITH ❤️ PRIYANSHI SHUKLA
