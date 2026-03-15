"""
app.py — Streamlit UI cho hệ thống RAG (FAISS + FastEmbed + GPT-4.1-mini)

Chạy:  streamlit run app.py
"""

import os
import io
import json
import contextlib
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from Ingest_Local_Fastest import (
    offline_ingest,
    list_ingested_documents,
    delete_document,
    load_documents_store,
    DEFAULT_MODEL as INGEST_DEFAULT_MODEL,
)
from Query_Local_Fastest import RAGQueryEngine, rag_query

# ======================================================
# CONSTANTS
# ======================================================

DATA_DIR      = "data"
VECTOR_DIR    = "vector_store"
HISTORY_FILE  = os.path.join(VECTOR_DIR, "streamlit_history.json")
DEFAULT_MODEL = INGEST_DEFAULT_MODEL

os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="RAG Notebook", page_icon="📚",
    layout="wide", initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0f1117; color: #e8e8e8; }
h1 { font-family: 'DM Serif Display', serif; font-size: 2rem !important; color: #f0e6d3 !important; }
h2 { font-family: 'DM Serif Display', serif; color: #d4c4a8 !important; font-size: 1.3rem !important; }
h3 { font-size: 0.95rem !important; color: #a89880 !important; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }
[data-testid="stSidebar"] { background: #0a0c11 !important; border-right: 1px solid #1e2330; }
.answer-box { background: #131820; border-left: 3px solid #3a7bd5; border-radius: 0 10px 10px 0; padding: 20px 24px; margin: 12px 0; line-height: 1.75; }
.source-badge { display: inline-block; background: #1e2a3a; border: 1px solid #2d4060; border-radius: 6px; padding: 3px 10px; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #7eb8f7; margin: 3px 3px 3px 0; }
.cache-hit  { background:#0d2d1a; border-color:#1a5c34; color:#4caf82; }
.cache-miss { background:#2d1a0d; border-color:#5c3a1a; color:#cf8f4c; }
.chunk-preview { background: #0d1117; border: 1px solid #1e2330; border-radius: 8px; padding: 12px 16px; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #8899aa; margin: 6px 0; white-space: pre-wrap; max-height: 200px; overflow-y: auto; }
.log-box { background: #080b10; border: 1px solid #1a2030; border-radius: 8px; padding: 10px 14px; font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; color: #6a9f6a; max-height: 280px; overflow-y: auto; white-space: pre-wrap; }
.stButton > button { background: #1a2236 !important; border: 1px solid #2d4060 !important; color: #a8c4e8 !important; border-radius: 8px !important; transition: all 0.2s !important; }
.stButton > button:hover { background: #203050 !important; border-color: #3a7bd5 !important; color: #d0e8ff !important; }
.stTextArea textarea, .stTextInput input { background: #161b27 !important; border: 1px solid #2a3550 !important; color: #e8e8e8 !important; border-radius: 10px !important; }
hr { border-color: #1e2330 !important; }
[data-testid="stMetric"] { background: #161b27; border: 1px solid #1e2a3a; border-radius: 10px; padding: 12px 16px; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# SESSION STATE
# ======================================================

def init_state():
    for k, v in {
        "rag_engine": None, "history": [], "logs": "",
        "preview_doc": None, "active_answer": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ======================================================
# HELPERS
# ======================================================

def capture_logs(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()

def append_log(text: str):
    st.session_state["logs"] += text + "\n"

def load_history() -> List[Dict]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(history: List[Dict]):
    os.makedirs(VECTOR_DIR, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def faiss_exists() -> bool:
    return (
        os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")) and
        os.path.exists(os.path.join(VECTOR_DIR, "index.pkl"))
    )

def format_size(b: int) -> str:
    if b < 1024:       return f"{b} B"
    if b < 1024**2:    return f"{b/1024:.1f} KB"
    return f"{b/1024**2:.1f} MB"

def export_to_markdown(item: Dict) -> str:
    src = "\n".join(f"- {s['file_name']} | Page {s['page']} | Chunk {s['chunk_id']}" for s in item.get("sources", []))
    return f"# Q&A Export\n\n**Thời gian:** {item.get('ts','')}\n\n## Câu hỏi\n\n{item.get('question','')}\n\n## Câu trả lời\n\n{item.get('answer','')}\n\n## Nguồn\n\n{src}\n"

# ======================================================
# SIDEBAR
# ======================================================

def render_sidebar() -> Dict:
    with st.sidebar:
        st.markdown("## 📚 RAG Notebook")
        st.markdown("---")

        # ── Trạng thái ──
        st.markdown("### Trạng thái hệ thống")
        docs = list_ingested_documents(VECTOR_DIR)
        c1, c2 = st.columns(2)
        c1.metric("Tài liệu", len(docs))
        c2.metric("FAISS", "✅" if faiss_exists() else "❌")
        cache_size = 0
        cache_path = os.path.join(VECTOR_DIR, "query_cache.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f: cache_size = len(json.load(f))
            except Exception: pass
        st.metric("Cache entries", cache_size)

        st.markdown("---")

        # ── Cấu hình Query ──
        st.markdown("### ⚙️ Cấu hình Query")
        model_name  = st.text_input("Embedding model", value=DEFAULT_MODEL,
                                    help="Phải khớp với model dùng khi ingest")
        temperature = st.slider("Temperature",        0.0,  1.0,  0.30, 0.05)
        retrieve_k  = st.slider("Retrieve K (FAISS)", 5,    30,   15)
        top_n       = st.slider("Top N (vào LLM)",    1,    10,   5)
        use_rerank  = st.toggle("🔁 FlashRank Rerank", value=False)
        use_cache   = st.toggle("⚡ Semantic Cache",   value=True)

        st.markdown("---")

        # ── Cấu hình Cache ──
        st.markdown("### 🗄️ Cấu hình Cache")
        cache_threshold = st.slider(
            "Cache threshold", 0.50, 1.0, 0.80, 0.05,
            help="Similarity ≥ threshold → trả về từ cache",
            disabled=not use_cache,
        )
        cache_max_size = st.slider(
            "Cache max size", 10, 500, 200, 10,
            help="Số entry tối đa lưu trong cache",
            disabled=not use_cache,
        )

        st.markdown("---")

        # ── Cấu hình Chunking ──
        st.markdown("### ✂️ Cấu hình Chunking")
        st.caption("Áp dụng khi bấm **Ingest tài liệu**")
        chunk_size = st.slider(
            "Chunk size (chars)", 200, 1500, 600, 50,
            help="Độ dài tối đa mỗi raw chunk",
        )
        chunk_overlap = st.slider(
            "Chunk overlap (chars)", 0, 300, 150, 10,
            help="Số ký tự overlap giữa các chunk liền kề",
        )
        semantic_threshold = st.slider(
            "Semantic merge threshold", 0.50, 1.0, 0.75, 0.05,
            help="Similarity ≥ threshold → merge 2 chunk liền kề",
        )
        semantic_max_chars = st.slider(
            "Semantic max chars", 600, 3000, 1200, 100,
            help="Độ dài tối đa sau khi merge semantic",
        )

        st.markdown("---")

        # ── API Key ──
        st.markdown("### 🔑 API Key")
        if os.getenv("OPENAI_API_KEY"):
            st.success("OPENAI_API_KEY ✅ (từ .env)")
        else:
            key_in = st.text_input("OPENAI_API_KEY", type="password")
            if key_in:
                os.environ["OPENAI_API_KEY"] = key_in
                st.success("Key đã được set!")

        st.markdown("---")
        with st.expander("📖 Hướng dẫn chạy"):
            st.code("python -m venv .venv\n.venv\\Scripts\\activate\npip install -r requirements.txt\nstreamlit run app.py", language="bash")

        return {
            "model_name": model_name, "temperature": temperature,
            "retrieve_k": retrieve_k, "top_n": top_n,
            "use_rerank": use_rerank, "use_cache": use_cache,
            # Cache params
            "cache_threshold": cache_threshold, "cache_max_size": cache_max_size,
            # Chunking params
            "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
            "semantic_threshold": semantic_threshold, "semantic_max_chars": semantic_max_chars,
        }

# ======================================================
# TAB 1 — DOCUMENTS
# ======================================================

def render_documents_tab(config: Dict):
    st.markdown("## 🗂️ Quản lý tài liệu")
    st.markdown("### Thêm tài liệu mới")

    uploaded = st.file_uploader("Upload PDF", type=["pdf"],
                                 accept_multiple_files=True, label_visibility="collapsed")
    if uploaded:
        saved = []
        for uf in uploaded:
            with open(os.path.join(DATA_DIR, uf.name), "wb") as f:
                f.write(uf.getbuffer())
            saved.append(uf.name)
        st.success(f"Đã lưu {len(saved)} file: {', '.join(saved)}")

    # Tóm tắt config chunking đang áp dụng
    st.caption(
        f"Chunking: chunk_size=**{config['chunk_size']}** | "
        f"overlap=**{config['chunk_overlap']}** | "
        f"semantic_threshold=**{config['semantic_threshold']}** | "
        f"semantic_max_chars=**{config['semantic_max_chars']}**"
    )

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        do_ingest = st.button("📥 Ingest tài liệu", use_container_width=True)

    if do_ingest:
        log_ph    = st.empty()
        prog_bar  = st.progress(0)
        status_ph = st.empty()

        with st.spinner("Đang ingest..."):
            status_ph.info("⏳ Bắt đầu pipeline ingest...")
            prog_bar.progress(10)

            # Truyền đầy đủ chunking params từ sidebar
            result, logs = capture_logs(
                offline_ingest,
                data_dir=DATA_DIR,
                vector_dir=VECTOR_DIR,
                model_name=config["model_name"],
                chunk_size=config["chunk_size"],
                chunk_overlap=config["chunk_overlap"],
                semantic_threshold=config["semantic_threshold"],
                semantic_max_chars=config["semantic_max_chars"],
            )

            append_log(f"[{datetime.now().strftime('%H:%M:%S')}] INGEST\n{logs}")
            log_ph.markdown(f'<div class="log-box">{logs.replace(chr(10),"<br>")}</div>', unsafe_allow_html=True)
            prog_bar.progress(100)
            st.session_state["rag_engine"] = None  # reset engine sau ingest

            if result["status"] == "success":
                status_ph.success(
                    f"✅ Ingest thành công: {len(result['new_pdf_files'])} file mới | "
                    f"{result['total_semantic_chunks']} semantic chunks"
                )
            elif result["status"] == "no_new_files":
                status_ph.warning("⚠️ Không có file mới — tất cả đã được ingest.")
            else:
                status_ph.error("❌ Ingest thất bại. Xem logs bên dưới.")

    st.markdown("---")
    st.markdown("### Danh sách tài liệu đã ingest")
    docs = list_ingested_documents(VECTOR_DIR)

    if not docs:
        st.info("Chưa có tài liệu nào. Upload PDF và bấm 'Ingest tài liệu'.")
        return

    for doc in docs:
        c1, c2, c3, _ = st.columns([3, 1, 1, 1])
        with c1:
            st.markdown(f"**📄 {doc['file_name']}**")
            st.caption(
                f"Ingested: {doc['ingested_at'][:16]}  |  "
                f"{format_size(doc['file_size'])}  |  "
                f"{doc['total_pages']} pages  |  "
                f"{doc['total_semantic_chunks']} chunks"
            )
        with c2:
            if st.button("🔍 Preview", key=f"prev_{doc['file_name']}"):
                st.session_state["preview_doc"] = doc["file_name"]
        with c3:
            if st.button("🗑️ Xóa", key=f"del_{doc['file_name']}"):
                with st.spinner(f"Đang xóa {doc['file_name']}..."):
                    del_result, del_logs = capture_logs(
                        delete_document, file_name=doc["file_name"],
                        vector_dir=VECTOR_DIR, model_name=config["model_name"],
                    )
                    append_log(f"[{datetime.now().strftime('%H:%M:%S')}] DELETE {doc['file_name']}\n{del_logs}")
                    st.session_state["rag_engine"] = None
                    if del_result["status"] == "deleted":
                        st.success(f"✅ Đã xóa: {del_result['removed_registry_count']} registry, {del_result['removed_chunk_count']} chunks.")
                    else:
                        st.error(del_result.get("message", "Lỗi không xác định"))
                    st.rerun()

        st.markdown('<hr style="margin:6px 0;border-color:#1e2330">', unsafe_allow_html=True)

    # Preview panel
    preview_doc = st.session_state.get("preview_doc")
    if preview_doc:
        st.markdown(f"---\n### 🔍 Preview: `{preview_doc}`")
        doc_meta = next((d for d in docs if d["file_name"] == preview_doc), None)
        if doc_meta:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pages",           doc_meta["total_pages"])
            c2.metric("Raw chunks",      doc_meta["total_raw_chunks"])
            c3.metric("Semantic chunks", doc_meta["total_semantic_chunks"])
            c4.metric("Size",            format_size(doc_meta["file_size"]))

        store  = load_documents_store(VECTOR_DIR)
        chunks = [item for item in store.get("documents", [])
                  if item.get("metadata", {}).get("file_name") == preview_doc][:5]
        if chunks:
            st.markdown(f"**Hiển thị {len(chunks)} semantic chunks đầu:**")
            for i, chunk in enumerate(chunks):
                meta = chunk.get("metadata", {})
                with st.expander(f"Chunk {meta.get('semantic_chunk_id',i)} | Page {meta.get('page','?')} | raw_ids: {meta.get('raw_chunk_ids',[])}"):
                    st.markdown(f'<div class="chunk-preview">{chunk["page_content"][:600]}...</div>', unsafe_allow_html=True)

        pdf_path = os.path.join(DATA_DIR, preview_doc)
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button("⬇️ Tải xuống PDF", data=f, file_name=preview_doc, mime="application/pdf")

        if st.button("✖ Đóng preview"):
            st.session_state["preview_doc"] = None
            st.rerun()

# ======================================================
# TAB 2 — Q&A
# ======================================================

def render_qa_tab(config: Dict):
    st.markdown("## 💬 Đặt câu hỏi")

    if not faiss_exists():
        st.warning("⚠️ Chưa có FAISS index. Vui lòng ingest tài liệu trước tại tab **Tài liệu**.")
        return
    if not os.getenv("OPENAI_API_KEY"):
        st.error("❌ Thiếu OPENAI_API_KEY. Vui lòng thêm vào file `.env` hoặc nhập ở sidebar.")
        return

    question = st.text_area("Câu hỏi", height=120,
                             placeholder="Nhập câu hỏi về tài liệu đã ingest...",
                             label_visibility="collapsed")

    col_submit, col_info = st.columns([1, 3])
    with col_submit:
        do_query = st.button("🔍 Gửi câu hỏi", use_container_width=True, type="primary")
    with col_info:
        st.caption(
            f"K={config['retrieve_k']} | N={config['top_n']} | "
            f"Rerank={'✅' if config['use_rerank'] else '❌'} | "
            f"Cache={'✅' if config['use_cache'] else '❌'} "
            f"(thr={config['cache_threshold']}, max={config['cache_max_size']})"
        )

    if do_query and question.strip():
        log_ph = st.empty()
        with st.spinner("🤔 Đang tìm kiếm và tổng hợp..."):
            if st.session_state["rag_engine"] is None:
                engine, _ = capture_logs(
                    lambda: RAGQueryEngine(
                        vector_dir=VECTOR_DIR,
                        model_name=config["model_name"],
                        retrieve_k=config["retrieve_k"],
                        top_n=config["top_n"],
                        use_rerank=config["use_rerank"],
                        use_cache=config["use_cache"],
                        cache_threshold=config["cache_threshold"],
                    )
                )
                st.session_state["rag_engine"] = engine
            else:
                engine = st.session_state["rag_engine"]

            if engine is None:
                st.error("Không thể khởi tạo engine.")
            else:
                result, query_logs = capture_logs(
                    engine.query, question=question,
                    use_rerank=config["use_rerank"],
                    use_cache=config["use_cache"],
                    verbose=True,
                )

                append_log(f"[{datetime.now().strftime('%H:%M:%S')}] QUERY: {question[:60]}\n{query_logs}")
                log_ph.markdown(f'<div class="log-box">{query_logs.replace(chr(10),"<br>")}</div>', unsafe_allow_html=True)

                history_item = {
                    "question": question, "answer": result["answer"],
                    "sources": result["sources"], "context": result["context"],
                    "from_cache": result["from_cache"],
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                st.session_state["history"].insert(0, history_item)
                save_history(st.session_state["history"])
                st.session_state["active_answer"] = history_item

    active = st.session_state.get("active_answer")
    if active:
        _render_answer_card(active)


def _render_answer_card(item: Dict):
    st.markdown("---")
    cache_class = "cache-hit" if item["from_cache"] else "cache-miss"
    cache_label = "⚡ Từ Cache" if item["from_cache"] else "🤖 Được tạo mới"
    st.markdown(f'<span class="source-badge {cache_class}">{cache_label}</span>', unsafe_allow_html=True)

    st.markdown("### Câu trả lời")
    st.markdown(f'<div class="answer-box">{item["answer"]}</div>', unsafe_allow_html=True)

    if item.get("sources"):
        st.markdown("**📎 Nguồn tham khảo:**")
        badges = " ".join(
            f'<span class="source-badge" title="Chunk {s["chunk_id"]}">📄 {s["file_name"]} · p.{s["page"]}</span>'
            for s in item["sources"]
        )
        st.markdown(badges, unsafe_allow_html=True)
        citation = "  |  ".join(
            f"Source: {s['file_name']}, Page: {s['page']}, Chunk: {s['chunk_id']}"
            for s in item["sources"]
        )
        st.code(citation, language=None)

    if item.get("context"):
        with st.expander("📋 Xem context đã dùng"):
            st.markdown(f'<div class="chunk-preview">{item["context"]}</div>', unsafe_allow_html=True)

    st.download_button(
        "⬇️ Export Markdown",
        data=export_to_markdown(item).encode("utf-8"),
        file_name=f"qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
    )

# ======================================================
# TAB 3 — HISTORY
# ======================================================

def render_history_tab():
    st.markdown("## 🕐 Lịch sử câu hỏi")
    if not st.session_state["history"]:
        st.session_state["history"] = load_history()

    history = st.session_state["history"]
    if not history:
        st.info("Chưa có câu hỏi nào trong lịch sử.")
        return

    c1, c2 = st.columns([1, 1])
    with c1: st.caption(f"Tổng: {len(history)} câu hỏi")
    with c2:
        if st.button("🗑️ Xóa toàn bộ history"):
            st.session_state["history"] = []
            save_history([])
            st.session_state["active_answer"] = None
            st.rerun()

    st.markdown("---")
    for i, item in enumerate(history):
        icon = "⚡" if item.get("from_cache") else "🤖"
        q    = item.get("question","")[:80] + ("..." if len(item.get("question","")) > 80 else "")
        with st.expander(f"{icon} [{item.get('ts','')}] {q}"):
            st.markdown(f"**❓ Câu hỏi:**\n\n{item['question']}")
            st.markdown("---")
            st.markdown(f'<div class="answer-box">{item["answer"]}</div>', unsafe_allow_html=True)
            if item.get("sources"):
                badges = " ".join(f'<span class="source-badge">📄 {s["file_name"]} · p.{s["page"]}</span>' for s in item["sources"])
                st.markdown(badges, unsafe_allow_html=True)
            ca, cb = st.columns(2)
            with ca:
                if st.button("📌 Xem đầy đủ", key=f"view_hist_{i}"):
                    st.session_state["active_answer"] = item
            with cb:
                st.download_button("⬇️ Export MD", data=export_to_markdown(item).encode("utf-8"),
                                   file_name=f"qa_{i}.md", mime="text/markdown", key=f"exp_{i}")

# ======================================================
# TAB 4 — LOGS
# ======================================================

def render_logs_tab():
    st.markdown("## 🪵 System Logs")
    c1, _ = st.columns([1, 5])
    with c1:
        if st.button("🧹 Clear logs"):
            st.session_state["logs"] = ""
            st.rerun()
    logs = st.session_state.get("logs", "")
    if logs:
        st.markdown(f'<div class="log-box" style="max-height:600px">{logs.replace(chr(10),"<br>")}</div>', unsafe_allow_html=True)
    else:
        st.info("Chưa có logs. Thực hiện ingest hoặc query để xem logs.")

# ======================================================
# MAIN
# ======================================================

def main():
    config = render_sidebar()

    st.markdown('<h1>📚 RAG Notebook</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6a8ab0;font-size:0.9rem;margin-top:-12px">FAISS · FastEmbed · FlashRank · GPT-4.1-mini</p>', unsafe_allow_html=True)

    tab_docs, tab_qa, tab_hist, tab_logs = st.tabs(["📁 Tài liệu","💬 Hỏi & Đáp","🕐 Lịch sử","🪵 Logs"])
    with tab_docs: render_documents_tab(config)
    with tab_qa:   render_qa_tab(config)
    with tab_hist: render_history_tab()
    with tab_logs: render_logs_tab()

if __name__ == "__main__":
    main()
