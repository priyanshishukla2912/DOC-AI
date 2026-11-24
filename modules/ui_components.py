"""
DocuChat - Modern UI Styling & Custom Streamlit Components
Author: Priyanshi Shukla
Description: Glassmorphism CSS, responsive metrics cards, citation badges,
security status banners, and official brand footer.
"""

import streamlit as st


CUSTOM_CSS = """
<style>
/* Main Container Styling */
.main .block-container {
    padding-top: 1.8rem;
    padding-bottom: 3.5rem;
    max-width: 1200px;
}

/* Header & Brand Banner */
.docuchat-header {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4338ca 100%);
    border-radius: 16px;
    padding: 24px 30px;
    color: white;
    box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.25);
    margin-bottom: 24px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.docuchat-header h1 {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
    color: #ffffff !important;
}

.docuchat-header p {
    font-size: 1.05rem;
    color: #c7d2fe;
    margin-top: 6px;
    margin-bottom: 12px;
}

.badge-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.badge-security {
    background-color: rgba(16, 185, 129, 0.2);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.4);
}

.badge-rag {
    background-color: rgba(99, 102, 241, 0.2);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.4);
}

.badge-gemini {
    background-color: rgba(245, 158, 11, 0.2);
    color: #fcd34d;
    border: 1px solid rgba(245, 158, 11, 0.4);
}

/* Metric & Feature Cards */
.metric-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 14px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-card:hover {
    border-color: rgba(99, 102, 241, 0.4);
    transform: translateY(-2px);
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #6366f1;
}

.metric-label {
    font-size: 0.85rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Citation & Source Box */
.citation-box {
    background-color: rgba(15, 23, 42, 0.6);
    border-left: 4px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-top: 10px;
    margin-bottom: 14px;
    font-size: 0.9rem;
    color: #cbd5e1;
}

.citation-meta {
    font-weight: 600;
    color: #818cf8;
    font-size: 0.82rem;
    margin-bottom: 4px;
}

/* Security Vault Card */
.vault-panel {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(52, 211, 153, 0.3);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}

/* Suggested Question Buttons */
div.stButton > button.suggested-btn {
    text-align: left;
    width: 100%;
    border-radius: 10px;
    background-color: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.2);
    color: #e2e8f0;
    padding: 8px 14px;
    font-size: 0.88rem;
    transition: all 0.2s ease;
}

div.stButton > button.suggested-btn:hover {
    background-color: rgba(99, 102, 241, 0.2);
    border-color: #6366f1;
    color: #ffffff;
}

/* Clean Footer */
.docuchat-footer {
    text-align: center;
    padding: 30px 15px 15px 15px;
    margin-top: 45px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.footer-signature {
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: #f43f5e;
    text-shadow: 0 0 12px rgba(244, 63, 94, 0.3);
    display: inline-block;
    padding: 8px 18px;
    border-radius: 9999px;
    background: rgba(244, 63, 94, 0.08);
    border: 1px solid rgba(244, 63, 94, 0.25);
}

.footer-sub {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-top: 8px;
}
</style>
"""


def apply_custom_styles():
    """Injects high-end CSS into the Streamlit session."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header():
    """Renders the main hero banner."""
    st.markdown("""
    <div class="docuchat-header">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h1>DocuChat <span style="font-size: 1.4rem; opacity: 0.85;">🔒</span></h1>
                <p>AI-Powered Document Q&A System with Client-Side Encryption & Grounded RAG Pipeline</p>
            </div>
            <div>
                <span class="badge-pill badge-security">🔒 AES-256 Encrypted</span>
                <span class="badge-pill badge-rag">⚡ LangChain RAG</span>
                <span class="badge-pill badge-gemini">✨ Google Gemini</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_footer():
    """
    Renders the official footer at the bottom of the interface.
    Explicitly satisfies requirement: 'MADE WITH HEART EMOJJI PRIYANSHI SHUKLA'.
    """
    st.markdown("""
    <div class="docuchat-footer">
        <div class="footer-signature">
            MADE WITH ❤️ PRIYANSHI SHUKLA
        </div>
        <div class="footer-sub">
            DocuChat • Production-Grade RAG Pipeline • Zero-Knowledge Cryptographic Ingestion
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_citation_card(source_name: str, page_num: int, score: int, text_snippet: str):
    """Renders a styled citation card."""
    badge_color = "#10b981" if score >= 70 else "#f59e0b" if score >= 40 else "#64748b"
    st.markdown(f"""
    <div class="citation-box">
        <div class="citation-meta">
            📄 <strong>{source_name}</strong> • Page/Section <strong>{page_num}</strong> • 
            <span style="color: {badge_color};">Relevance {score}%</span>
        </div>
        <div>{text_snippet}</div>
    </div>
    """, unsafe_allow_html=True)
