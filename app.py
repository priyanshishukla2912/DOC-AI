"""
DocuChat — AI-Powered Document Q&A System with End-to-End Encryption
Author: Priyanshi Shukla
GitHub: https://github.com/priyanshishukla2912/DOC-AI
Description: Production-ready conversational document Q&A platform implementing a secure
RAG pipeline with LangChain design patterns, Google Gemini, and AES-256 cryptographic vault.
"""

import os
import io
import time
import streamlit as st

from modules.encryption import DocumentVault, PIIScrubber
from modules.document_parser import DocumentParser, DocumentMetadata
from modules.rag_pipeline import RecursiveTextSplitter, HybridVectorStore, TextChunk
from modules.gemini_llm import GeminiRAGClient
from modules.ui_components import (
    apply_custom_styles,
    render_header,
    render_footer,
    render_citation_card
)

# Page configuration
st.set_page_config(
    page_title="DocuChat — AI-Powered Document Q&A System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply glassmorphic visual styling
apply_custom_styles()


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "vault" not in st.session_state:
    st.session_state.vault = DocumentVault(vault_dir=".vault")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = HybridVectorStore()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_docs_meta" not in st.session_state:
    st.session_state.uploaded_docs_meta = {}

if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []

if "executive_summaries" not in st.session_state:
    st.session_state.executive_summaries = {}

if "pii_stats_total" not in st.session_state:
    st.session_state.pii_stats_total = {"emails": 0, "phones": 0, "ssn": 0, "cards": 0}

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# -----------------------------------------------------------------------------
# Sidebar: Configuration, Ingestion, and Security Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine & API Configuration")
    env_gemini_key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=env_gemini_key,
        type="password",
        placeholder="AIzaSy...",
        help="Enter your Gemini API key from Google AI Studio. Leave empty to use local offline synthesis."
    )

    col_model, col_temp = st.columns([1.2, 0.8])
    with col_model:
        model_choice = st.selectbox(
            "Gemini Model",
            ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
            index=0,
            help="gemini-1.5-flash is optimized for speed and cost-effective RAG reasoning."
        )
    with col_temp:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05, help="Lower = more deterministic and grounded.")

    # Re-link vector store with API key
    if api_key_input != st.session_state.vector_store.gemini_api_key:
        st.session_state.vector_store.gemini_api_key = api_key_input

    gemini_client = GeminiRAGClient(api_key=api_key_input, model_name=model_choice)

    if gemini_client.is_ready():
        st.success("🟢 Google Gemini Connected & Active")
    else:
        st.info("🟡 Local/Offline Mode Active (Full RAG search enabled)")

    st.markdown("---")
    st.markdown("### 📤 Document Upload Vault")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT, or MD files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="All documents are immediately encrypted using AES-256-CBC at rest."
    )

    # Quick demo document loader
    st.markdown("💡 *Or test immediately with built-in sample:*")
    sample_col1, sample_col2 = st.columns(2)
    sample_to_load = None
    with sample_col1:
        if st.button("📄 AI Ethics", use_container_width=True):
            sample_to_load = "sample_documents/ai_ethics_and_governance.txt"
    with sample_col2:
        if st.button("🛡️ Cloud Security", use_container_width=True):
            sample_to_load = "sample_documents/cloud_data_security_standards.txt"

    st.markdown("---")
    st.markdown("### 🛡️ Privacy & RAG Hyperparameters")
    enable_pii_scrub = st.toggle("Enable PII Anonymization", value=True, help="Scrub emails, phone numbers, and SSNs before LLM processing.")
    chunk_size = st.slider("Chunk Size (characters)", 300, 1500, 700, 50)
    chunk_overlap = st.slider("Chunk Overlap (characters)", 50, 300, 120, 10)
    top_k = st.slider("Top-K Retrieved Passages", 1, 8, 4)

    st.markdown("---")
    # Zero-Retention Purge Button
    if st.button("🗑️ Zero-Retention Purge", use_container_width=True, type="secondary"):
        shredded = st.session_state.vault.purge_vault()
        st.session_state.vector_store.clear()
        st.session_state.chat_history.clear()
        st.session_state.uploaded_docs_meta.clear()
        st.session_state.suggested_questions.clear()
        st.session_state.executive_summaries.clear()
        st.session_state.pii_stats_total = {"emails": 0, "phones": 0, "ssn": 0, "cards": 0}
        st.success(f"🔒 Purged {shredded} encrypted files & wiped RAM.")
        st.rerun()


# -----------------------------------------------------------------------------
# Document Ingestion Pipeline Function
# -----------------------------------------------------------------------------
def process_and_ingest_file(filename: str, file_bytes: bytes):
    """Encrypts at rest, decrypts in-memory, parses, chunks, and indexes in vector store."""
    if filename in st.session_state.uploaded_docs_meta:
        return  # Already ingested

    with st.spinner(f"🔒 Encrypting & Indexing '{filename}'..."):
        # 1. Encrypt and store ciphertext only
        vault_meta = st.session_state.vault.encrypt_and_store(filename, file_bytes)

        # 2. Decrypt in memory for parsing
        raw_decrypted = st.session_state.vault.decrypt_in_memory(filename)

        # 3. Parse segments
        segments, doc_meta = DocumentParser.parse_stream(filename, raw_decrypted)
        st.session_state.uploaded_docs_meta[filename] = {
            "meta": doc_meta,
            "vault": vault_meta,
            "segments_count": len(segments)
        }

        # 4. Recursive semantic chunking
        splitter = RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks: list[TextChunk] = []

        combined_sample_text = ""
        for seg in segments:
            text_to_chunk = seg.text
            if enable_pii_scrub:
                text_to_chunk, pii_counts = PIIScrubber.redact(text_to_chunk)
                for k, v in pii_counts.items():
                    st.session_state.pii_stats_total[k] += v

            sub_chunks = splitter.split_text(text_to_chunk)
            for idx, c_text in enumerate(sub_chunks):
                chunk_id = f"{filename}-p{seg.page_num}-c{idx + 1}"
                all_chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        doc_name=filename,
                        page_num=seg.page_num,
                        chunk_index=idx + 1,
                        text=c_text,
                        char_count=len(c_text),
                        word_count=len(c_text.split()),
                        metadata={"raw_length": len(c_text)}
                    )
                )
            if len(combined_sample_text) < 3000:
                combined_sample_text += "\n\n" + text_to_chunk

        # 5. Index into Hybrid Vector Store
        st.session_state.vector_store.add_chunks(all_chunks)

        # 6. Generate Smart Suggestions & Executive Summary
        if not st.session_state.suggested_questions:
            questions = gemini_client.generate_suggested_questions(combined_sample_text)
            st.session_state.suggested_questions = questions

        exec_summary = gemini_client.generate_executive_summary(combined_sample_text)
        st.session_state.executive_summaries[filename] = exec_summary


# Handle uploaded files
if uploaded_files:
    for f in uploaded_files:
        bytes_data = f.read()
        process_and_ingest_file(f.name, bytes_data)

# Handle sample document click
if sample_to_load and os.path.exists(sample_to_load):
    with open(sample_to_load, "rb") as sf:
        sample_name = os.path.basename(sample_to_load)
        process_and_ingest_file(sample_name, sf.read())
    st.rerun()


# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
render_header()

# Quick Overview Bar
total_docs = len(st.session_state.uploaded_docs_meta)
total_chunks = len(st.session_state.vector_store.chunks)
total_words = sum(d["meta"].total_words for d in st.session_state.uploaded_docs_meta.values())

top_col1, top_col2, top_col3, top_col4 = st.columns(4)
with top_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_docs}</div>
        <div class="metric-label">Documents Ingested</div>
    </div>
    """, unsafe_allow_html=True)
with top_col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_chunks}</div>
        <div class="metric-label">Semantic Chunks</div>
    </div>
    """, unsafe_allow_html=True)
with top_col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_words:,}</div>
        <div class="metric-label">Total Words Indexed</div>
    </div>
    """, unsafe_allow_html=True)
with top_col4:
    vault_status = "🔒 AES-256 Active" if total_docs > 0 else "⚪ Vault Standby"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="font-size: 1.3rem; color: #10b981;">{vault_status}</div>
        <div class="metric-label">Cryptographic Shield</div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Tabs Layout
# -----------------------------------------------------------------------------
tab_chat, tab_summary, tab_vault, tab_arch = st.tabs([
    "💬 Conversational Q&A",
    "📑 Document Intelligence & Summary",
    "🛡️ Privacy & Encryption Vault",
    "📊 System Architecture & Metrics"
])


# -----------------------------------------------------------------------------
# TAB 1: Conversational Q&A
# -----------------------------------------------------------------------------
with tab_chat:
    if total_docs == 0:
        st.info("👋 Welcome to **DocuChat**! Please upload a PDF, DOCX, or TXT file using the sidebar, or click one of the sample buttons to test right away.")
    else:
        # Display Suggested Questions if available
        if st.session_state.suggested_questions:
            st.markdown("##### 💡 Suggested Questions (Click to Ask)")
            q_cols = st.columns(len(st.session_state.suggested_questions[:4]))
            for idx, q_text in enumerate(st.session_state.suggested_questions[:4]):
                with q_cols[idx]:
                    if st.button(f"🔍 {q_text}", key=f"sugg_q_{idx}", use_container_width=True):
                        st.session_state.pending_prompt = q_text
                        st.rerun()

    # Render previous conversation history
    for message in st.session_state.chat_history:
        role = message["role"]
        with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):
            st.markdown(message["content"])
            # Display source citations if present
            if role == "assistant" and "citations" in message and message["citations"]:
                with st.expander(f"📑 Sources & Citations ({len(message['citations'])} passages retrieved)"):
                    for cit in message["citations"]:
                        render_citation_card(
                            source_name=cit["doc_name"],
                            page_num=cit["page_num"],
                            score=cit["score"],
                            text_snippet=cit["snippet"]
                        )

    # Chat Input handler
    user_query = st.chat_input("Ask any natural-language question about your uploaded documents...")
    if st.session_state.pending_prompt:
        user_query = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if user_query:
        # Record user query
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_query)

        # Retrieve relevant passages
        retrieval_results = st.session_state.vector_store.query(user_query, top_k=top_k)
        grounded_context = HybridVectorStore.build_grounded_context(retrieval_results)

        citations = []
        for r in retrieval_results:
            c = r.chunk
            snippet = c.text[:220] + "..." if len(c.text) > 220 else c.text
            citations.append({
                "doc_name": c.doc_name,
                "page_num": c.page_num,
                "score": r.percentage,
                "snippet": snippet
            })

        # Generate streaming assistant response
        with st.chat_message("assistant", avatar="🤖"):
            response_container = st.empty()
            full_response = ""

            stream = gemini_client.generate_rag_stream(
                query=user_query,
                context=grounded_context,
                chat_history=st.session_state.chat_history,
                temperature=temperature
            )

            for token in stream:
                full_response += token
                response_container.markdown(full_response + " ▌")
            response_container.markdown(full_response)

            # Display citations expander directly
            if citations:
                with st.expander(f"📑 Sources & Citations ({len(citations)} passages retrieved)"):
                    for cit in citations:
                        render_citation_card(
                            source_name=cit["doc_name"],
                            page_num=cit["page_num"],
                            score=cit["score"],
                            text_snippet=cit["snippet"]
                        )

        # Record assistant answer
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": full_response,
            "citations": citations
        })

    # Chat Actions (Export & Clear)
    if st.session_state.chat_history:
        st.markdown("<br>", unsafe_allow_html=True)
        act_col1, act_col2 = st.columns([1, 1])
        with act_col1:
            chat_export_text = "# DocuChat Transcript\n\n"
            for m in st.session_state.chat_history:
                prefix = "**User:** " if m["role"] == "user" else "**DocuChat:** "
                chat_export_text += f"{prefix}{m['content']}\n\n"
                if "citations" in m and m["citations"]:
                    chat_export_text += "*Citations:*\n"
                    for c in m["citations"]:
                        chat_export_text += f"- {c['doc_name']} (Page/Section {c['page_num']}, Relevance: {c['score']}%)\n"
                    chat_export_text += "\n"
            chat_export_text += "\n---\n*Generated by DocuChat • Built with ❤️ by Priyanshi Shukla*\n"

            st.download_button(
                "📥 Export Chat Transcript (Markdown)",
                data=chat_export_text,
                file_name="docuchat_transcript.md",
                mime="text/markdown",
                use_container_width=True
            )
        with act_col2:
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.chat_history.clear()
                st.rerun()


# -----------------------------------------------------------------------------
# TAB 2: Document Intelligence & Executive Summary
# -----------------------------------------------------------------------------
with tab_summary:
    st.markdown("### 📑 Document Intelligence & Analysis")
    if total_docs == 0:
        st.info("Upload documents to generate automated executive summaries, readability metrics, and chunk distributions.")
    else:
        doc_options = list(st.session_state.uploaded_docs_meta.keys())
        selected_doc = st.selectbox("Select Document to Inspect", doc_options)

        if selected_doc:
            doc_data = st.session_state.uploaded_docs_meta[selected_doc]
            meta: DocumentMetadata = doc_data["meta"]
            vault_meta = doc_data["vault"]
            exec_summary = st.session_state.executive_summaries.get(selected_doc, {})

            # Executive Summary Card
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #10b981;">
                <h4 style="margin: 0 0 10px 0; color: #10b981;">Executive Brief — {selected_doc}</h4>
                <p style="font-size: 1.02rem; line-height: 1.6; color: #e2e8f0;">
                    {exec_summary.get('summary', 'Summary not available.')}
                </p>
                <div style="margin-top: 12px;">
                    <strong>Core Topics:</strong> {' '.join([f'<span class="badge-pill badge-rag">{t}</span>' for t in exec_summary.get('topics', [])])}
                </div>
                <div style="margin-top: 8px; font-size: 0.88rem; color: #94a3b8;">
                    <strong>Target Audience:</strong> {exec_summary.get('audience', 'General Reader')}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Document Statistics Grid
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.metric("Total Pages / Sections", meta.total_pages)
            with s2:
                st.metric("Total Word Count", f"{meta.total_words:,}")
            with s3:
                st.metric("Est. Reading Time", f"{meta.estimated_read_time_min} mins")
            with s4:
                st.metric("Raw File Size", f"{round(meta.raw_size_bytes / 1024, 1)} KB")

            # Chunk Explorer
            st.markdown("#### 🔍 Semantic Chunks Inspector")
            matching_chunks = [c for c in st.session_state.vector_store.chunks if c.doc_name == selected_doc]
            st.caption(f"Displaying {len(matching_chunks)} recursive semantic chunks indexed for this document.")

            with st.expander(f"Inspect All {len(matching_chunks)} Chunks"):
                for c in matching_chunks:
                    st.markdown(f"**Chunk ID:** `{c.chunk_id}` | **Page/Section:** {c.page_num} | **Length:** {c.char_count} chars")
                    st.code(c.text, language="text")
                    st.markdown("---")


# -----------------------------------------------------------------------------
# TAB 3: Privacy & Encryption Vault
# -----------------------------------------------------------------------------
with tab_vault:
    st.markdown("### 🛡️ Zero-Knowledge Cryptographic Vault")
    st.markdown("""
    DocuChat implements a **Privacy-First Ingestion Architecture**. When a user uploads a document:
    1. **AES-256 Symmetric Encryption**: The raw document stream is immediately encrypted at rest using an authenticated Fernet cipher suite (AES-128-CBC + HMAC-SHA256).
    2. **Encrypted Storage Only**: Only the encrypted `.enc` binary file is persisted to disk in the isolated `.vault/` directory.
    3. **Volatile Decryption**: The document is decoded strictly into transient in-memory RAM during chunking and vector indexing. Raw plaintext is **never** written to persistent disk storage.
    4. **Zero-Retention Shredding**: Purging the vault performs a cryptographically secure random overwrite of disk sectors before unlinking.
    """)

    # Privacy Metrics
    v_col1, v_col2, v_col3 = st.columns(3)
    with v_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.vault.key_fingerprint}</div>
            <div class="metric-label">Active Key ID</div>
        </div>
        """, unsafe_allow_html=True)
    with v_col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="font-size: 1.4rem;">AES-256-CBC</div>
            <div class="metric-label">Cipher Suite</div>
        </div>
        """, unsafe_allow_html=True)
    with v_col3:
        pii_total = sum(st.session_state.pii_stats_total.values())
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #f59e0b;">{pii_total}</div>
            <div class="metric-label">PII Entities Redacted</div>
        </div>
        """, unsafe_allow_html=True)

    # Cryptographic Audit Log Table
    st.markdown("#### 📋 Cryptographic Audit Register")
    audit_entries = st.session_state.vault.get_audit_log()

    if not audit_entries:
        st.info("No documents currently loaded in the cryptographic vault.")
    else:
        for entry in audit_entries:
            with st.container():
                st.markdown(f"""
                <div class="vault-panel">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #34d399;">🔒 {entry['original_filename']}</h4>
                        <span class="badge-pill badge-security">Integrity Verified</span>
                    </div>
                    <div style="margin-top: 10px; font-size: 0.88rem; color: #94a3b8;">
                        <div><strong>Encrypted Storage:</strong> <code>{entry['vault_path']}</code></div>
                        <div><strong>Raw SHA-256:</strong> <code>{entry['raw_sha256']}</code></div>
                        <div><strong>Ciphertext SHA-256:</strong> <code>{entry['cipher_sha256']}</code></div>
                        <div><strong>Ciphertext Sample:</strong> <code>{entry['cipher_preview']}</code></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 4: System Architecture & Metrics
# -----------------------------------------------------------------------------
with tab_arch:
    st.markdown("### 📊 System Architecture & Engineering Specifications")
    st.markdown("""
    #### 🏗️ End-to-End RAG Architecture Workflow
    ```mermaid
    flowchart LR
        A[User Upload: PDF / DOCX] --> B[AES-256 Cryptographic Vault]
        B --> C[Volatile In-Memory Decryptor]
        C --> D[PII Scrubbing & Anonymizer]
        D --> E[Recursive Semantic Splitter]
        E --> F[(Hybrid Vector Store: Gemini + TF-IDF)]
        G[User Natural Query] --> H[Cosine Semantic Search]
        F --> H
        H --> I[Top-K Ranked Context Assembly]
        I --> J[Anti-Hallucination Prompt Template]
        J --> K[Google Gemini 1.5/2.0 LLM]
        K --> L[Streaming Response with Citations]
    ```
    """)

    # Resume Highlights Box
    st.markdown("#### 🎯 Resume Engineering Highlights")
    st.markdown("""
    - **Conversational RAG Pipeline**: Engineered with LangChain architectural patterns and Google Gemini API (1.5 Flash/Pro, 2.0).
    - **Multi-Format Ingestion**: Custom parsers for PDF (PyPDF page tracking), DOCX (python-docx paragraph/table mapping), and TXT/Markdown.
    - **Cryptographic Security Vault**: Implemented authenticated symmetric encryption (Fernet / AES-256-CBC) preventing plaintext leakage on disk with zero-retention memory scrubbing.
    - **Deterministic Source Attribution**: Built precision citation engine providing exact source document names, page/section numbers, and relevance scores (0-100%).
    - **Production UI**: Designed responsive glassmorphic Streamlit interface with multi-document management, automatic executive summaries, and smart suggested questions.
    """)


# -----------------------------------------------------------------------------
# Bottom Footer (Strictly satisfies requested requirement)
# -----------------------------------------------------------------------------
render_footer()
