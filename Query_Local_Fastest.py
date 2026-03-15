"""
query.py — RAG Query Pipeline (FAISS + FlashRank Rerank + Semantic Cache + GPT-4.1-mini)

Pipeline chuẩn RAG + Rerank + Cache:
  1. Load FAISS index đã build bởi ingest.py
  2. Semantic Cache — nếu câu hỏi tương tự đã hỏi rồi → trả về ngay
  3. Embed câu hỏi bằng FastEmbedEmbeddings (BAAI/bge-small-en-v1.5)
  4. Retrieve top-K candidates từ FAISS
  5. FlashRank Rerank → chọn ra top-N chunk liên quan nhất (optional, local)
  6. Ghép context → build prompt
  7. Gửi sang ChatOpenAI (gpt-4.1-mini) → trả lời
  8. Lưu kết quả vào cache

Cần:
- vector_store/ đã được tạo bởi ingest.py
- File .env ở thư mục gốc project chứa:
    OPENAI_API_KEY=sk-...
"""

import os
import json
import time
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from flashrank import Ranker, RerankRequest
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


# ======================================================
# CONSTANTS
# ======================================================

DEFAULT_MODEL           = "BAAI/bge-small-en-v1.5"   # khớp với ingest.py
DEFAULT_VECTOR_DIR      = "vector_store"
DEFAULT_TOP_K           = 5       # số chunk đưa vào LLM (sau rerank)
DEFAULT_RETRIEVE_K      = 15      # số chunk lấy từ FAISS (trước rerank)
DEFAULT_TEMPERATURE     = 0.3
FLASHRANK_MODEL         = "ms-marco-MiniLM-L-12-v2"

# Cache
DEFAULT_CACHE_FILE      = "query_cache.json"  # lưu trong vector_store/
DEFAULT_CACHE_THRESHOLD = 0.8    # similarity >= 0.7 → coi là câu hỏi tương tự
DEFAULT_CACHE_MAX_SIZE  = 200    # giữ tối đa 200 entry, xóa cũ nhất nếu đầy


# ======================================================
# EMBEDDING — FastEmbed (khớp hoàn toàn với ingest.py)
# ======================================================

def get_embedding_model(model_name: str = DEFAULT_MODEL) -> FastEmbedEmbeddings:
    """Khởi tạo FastEmbedEmbeddings để dùng với LangChain FAISS."""
    print(f"[EMBED] Loading FastEmbedEmbeddings: {model_name}")
    model = FastEmbedEmbeddings(model_name=model_name)
    print("[EMBED] Model ready")
    return model


# ======================================================
# STEP 1 — LOAD FAISS (có auto-rebuild nếu thiếu index)
# ======================================================

def _rebuild_faiss(
    vector_dir: str = DEFAULT_VECTOR_DIR,
    model_name: str = DEFAULT_MODEL,
) -> Optional[FAISS]:
    """Rebuild FAISS từ documents_store.json khi thiếu index files."""
    doc_store_path = os.path.join(vector_dir, "documents_store.json")
    with open(doc_store_path, "r", encoding="utf-8") as f:
        store = json.load(f)

    raw_docs = store.get("documents", [])
    if not raw_docs:
        return None

    docs = [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in raw_docs
    ]

    print(f"[FAISS] Rebuilding from {len(docs)} chunks...")
    embedding_model = get_embedding_model(model_name)
    db = FAISS.from_documents(docs, embedding_model)
    db.save_local(vector_dir)
    print(f"[FAISS] Rebuilt và saved → {vector_dir}")
    return db


def load_faiss(
    vector_dir: str = DEFAULT_VECTOR_DIR,
    model_name: str = DEFAULT_MODEL,
) -> FAISS:
    """
    Load FAISS index. Tự động rebuild từ documents_store.json nếu thiếu index.
    """
    index_file = os.path.join(vector_dir, "index.faiss")
    store_file = os.path.join(vector_dir, "index.pkl")
    doc_store  = os.path.join(vector_dir, "documents_store.json")

    if not (os.path.exists(index_file) and os.path.exists(store_file)):
        if os.path.exists(doc_store):
            print("[FAISS] Thiếu index → tự động rebuild từ documents_store.json...")
            db = _rebuild_faiss(vector_dir=vector_dir, model_name=model_name)
            if db is None:
                raise RuntimeError(
                    "[FAISS] documents_store.json rỗng. Hãy chạy ingest.py trước."
                )
            return db
        raise FileNotFoundError(
            f"[FAISS] Không tìm thấy index tại '{vector_dir}'.\n"
            "Hãy chạy ingest.py trước để build vector store."
        )

    embedding_model = get_embedding_model(model_name)
    db = FAISS.load_local(
        folder_path=vector_dir,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )
    print(f"[FAISS] Loaded index from '{vector_dir}'")
    return db


# ======================================================
# STEP 2 — SEMANTIC CACHE
# ======================================================

class SemanticCache:
    """
    Cache lưu câu hỏi + embedding + answer.
    Khi có câu hỏi mới → so sánh cosine similarity với cache.
    Nếu similarity >= threshold → trả về cached answer ngay, không gọi LLM.

    Cache được lưu vào: vector_store/query_cache.json
    """

    def __init__(
        self,
        vector_dir: str = DEFAULT_VECTOR_DIR,
        model_name: str = DEFAULT_MODEL,
        threshold: float = DEFAULT_CACHE_THRESHOLD,
        max_size: int = DEFAULT_CACHE_MAX_SIZE,
        cache_file: str = DEFAULT_CACHE_FILE,
    ):
        self.cache_path = os.path.join(vector_dir, cache_file)
        self.threshold  = threshold
        self.max_size   = max_size
        self.entries: List[Dict[str, Any]] = []

        # Dùng chung embedding model với FAISS để nhất quán
        self._embed_model = get_embedding_model(model_name)

        self._load()

    # ---------- I/O ----------

    def _load(self):
        """Đọc cache từ file JSON nếu tồn tại."""
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self.entries = json.load(f)
            print(f"[CACHE] Loaded {len(self.entries)} entries từ {self.cache_path}")
        else:
            self.entries = []
            print(f"[CACHE] Khởi tạo cache mới → {self.cache_path}")

    def _save(self):
        """Lưu cache xuống file JSON."""
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)

    # ---------- Core logic ----------

    def _embed(self, text: str) -> List[float]:
        # embed_query() trả về List[float] → ép kiểu tường minh để chắc chắn
        return list(self._embed_model.embed_query(text))

    def lookup(self, query: str, verbose: bool = True) -> Optional[Dict[str, Any]]:
        """
        Tìm câu hỏi tương tự trong cache.
        Trả về cached result nếu similarity >= threshold, ngược lại trả về None.
        """
        if not self.entries:
            return None

        # Ép sang np.array để np.dot() hoạt động đúng
        query_emb = np.array(self._embed(query))

        best_score = -1.0
        best_entry = None

        for entry in self.entries:
            cached_emb = np.array(entry["embedding"])
            # Cosine similarity (đã normalize nên chỉ cần dot product)
            score = float(np.dot(query_emb, cached_emb))
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self.threshold:
            if verbose:
                print(f"[CACHE] ✅ HIT — similarity={best_score:.4f} (≥{self.threshold})")
                print(f"[CACHE]    Câu hỏi gốc: '{best_entry['query']}'")
            return best_entry["result"]

        if verbose:
            print(f"[CACHE] ❌ MISS — best similarity={best_score:.4f} (<{self.threshold})")
        return None

    def save_entry(self, query: str, result: Dict[str, Any]):
        """
        Lưu câu hỏi mới + kết quả vào cache.
        Xóa entry cũ nhất nếu cache đầy.
        """
        # Không lưu chunks (Document objects) vào cache vì không serializable
        cacheable_result = {
            "answer":  result["answer"],
            "sources": result["sources"],
            "context": result["context"],
        }

        entry = {
            "query":     query,
            "embedding": self._embed(query),
            "result":    cacheable_result,
            "cached_at": datetime.now().isoformat(),
        }

        self.entries.append(entry)

        # Xóa entry cũ nhất nếu vượt max_size
        if len(self.entries) > self.max_size:
            removed = self.entries.pop(0)
            print(f"[CACHE] Max size đạt {self.max_size} → xóa entry cũ nhất: '{removed['query'][:50]}'")

        self._save()
        print(f"[CACHE] Saved entry mới. Tổng: {len(self.entries)} entries")

    def clear(self):
        """Xóa toàn bộ cache."""
        self.entries = []
        self._save()
        print("[CACHE] Đã xóa toàn bộ cache.")

    def stats(self) -> Dict[str, Any]:
        """Trả về thông tin cache."""
        return {
            "total_entries": len(self.entries),
            "cache_file":    self.cache_path,
            "threshold":     self.threshold,
            "max_size":      self.max_size,
        }


# ======================================================
# STEP 3 — RETRIEVE
# ======================================================

def retrieve_chunks(
    db: FAISS,
    query: str,
    top_k: int = DEFAULT_RETRIEVE_K,
) -> List[Document]:
    """Similarity search: lấy top_k candidates từ FAISS."""
    results: List[tuple[Document, float]] = db.similarity_search_with_score(query, k=top_k)

    docs = []
    print(f"\n[RETRIEVE] {top_k} candidates từ FAISS:")
    for i, (doc, score) in enumerate(results):
        source = doc.metadata.get("file_name", "unknown")
        page   = doc.metadata.get("page", "?")
        print(f"  [{i+1:02d}] score={score:.4f} | {source} | page {page}")
        docs.append(doc)

    return docs


# ======================================================
# STEP 4 — FLASHRANK RERANK (local, không cần API key)
# ======================================================

def rerank_chunks(
    query: str,
    docs: List[Document],
    top_n: int = DEFAULT_TOP_K,
) -> List[Document]:
    """
    Dùng FlashRank (local) để chấm điểm lại candidates từ FAISS.
    Không cần API key, chạy hoàn toàn offline.
    Trả về top_n chunk có relevance score cao nhất.
    """
    ranker = Ranker(model_name=FLASHRANK_MODEL)

    # FlashRank yêu cầu đầu vào là list[dict] với id, text, meta
    passages = [
        {"id": i, "text": doc.page_content, "meta": doc.metadata}
        for i, doc in enumerate(docs)
    ]

    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)

    # Lấy top_n kết quả, map id về Document gốc
    reranked_docs = []
    print(f"\n[RERANK] FlashRank rerank {len(docs)} → top {top_n}:")
    for hit in results[:top_n]:
        doc    = docs[hit["id"]]
        source = doc.metadata.get("file_name", "unknown")
        page   = doc.metadata.get("page", "?")
        print(f"  [idx={hit['id']:02d}] score={hit['score']:.4f} | {source} | page {page}")
        reranked_docs.append(doc)

    return reranked_docs


# ======================================================
# STEP 5 — BUILD CONTEXT STRING
# ======================================================

def build_context(docs: List[Document]) -> str:
    """Ghép các chunk thành context string có header rõ ràng."""
    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("file_name", "unknown")
        page   = doc.metadata.get("page", "?")
        header = f"[Chunk {i+1} | Source: {source} | Page: {page}]"
        parts.append(f"{header}\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(parts)


# ======================================================
# STEP 6 — PROMPT TEMPLATE
# ======================================================

RAG_SYSTEM_PROMPT = """Bạn là một trợ lý nghiên cứu AI chuyên sâu, có nhiệm vụ trả lời câu hỏi DỰA TRÊN TÀI LIỆU được cung cấp.

## QUY TẮC BẮT BUỘC

1. **Chỉ dùng thông tin trong CONTEXT** — tuyệt đối không dùng kiến thức bên ngoài.
2. **Nếu không đủ thông tin** — trả lời đúng 1 câu: "Tài liệu được cung cấp không đề cập đến vấn đề này."
3. **Không suy diễn hoặc bịa đặt** — nếu không chắc, nói rõ "Tài liệu đề cập một phần..."
4. **Luôn trích dẫn nguồn** — ghi rõ (Source: tên_file, Page: số_trang) sau mỗi luận điểm.

## ĐỊNH DẠNG TRẢ LỜI

- Trả lời bằng **cùng ngôn ngữ với câu hỏi** (tiếng Việt hoặc tiếng Anh).
- Cấu trúc rõ ràng: dùng tiêu đề, gạch đầu dòng nếu có nhiều ý.
- Độ dài phù hợp: không quá ngắn (thiếu thông tin), không quá dài (lan man).
- Kết thúc bằng 1 câu tóm tắt ngắn nếu câu trả lời dài.
"""

RAG_HUMAN_PROMPT = """## CONTEXT (tài liệu tham khảo)

{context}

---

## CÂU HỎI

{question}

---

## YÊU CẦU TRẢ LỜI

Hãy suy nghĩ từng bước:
1. Xác định câu hỏi hỏi về điều gì.
2. Tìm thông tin liên quan trong CONTEXT.
3. Tổng hợp và trả lời có trích dẫn nguồn.

## TRẢ LỜI"""

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human",  RAG_HUMAN_PROMPT),
])


# ======================================================
# STEP 7 — GENERATE ANSWER
# ======================================================

def generate_answer(llm: ChatOpenAI, context: str, question: str) -> str:
    """Đưa context + question vào prompt → gọi LLM → trả về answer string."""
    chain = rag_prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content


# ======================================================
# HIGH-LEVEL API — hàm gọi nhanh 1 lần
# ======================================================

def rag_query(
    question: str,
    vector_dir: str  = DEFAULT_VECTOR_DIR,
    model_name: str  = DEFAULT_MODEL,
    retrieve_k: int  = DEFAULT_RETRIEVE_K,
    top_n: int       = DEFAULT_TOP_K,
    use_rerank: bool = False,              # False = tắt Rerank
    use_cache: bool  = True,              # True = bật Semantic Cache
    cache: Optional[SemanticCache] = None,
    db: Optional[FAISS]            = None,
    llm: Optional[ChatOpenAI]      = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Full RAG pipeline trong 1 hàm:
      question → [cache lookup] → retrieve(K) → [rerank(N)] → context → generate → [cache save]

    Params:
        question    : câu hỏi người dùng
        vector_dir  : thư mục chứa FAISS index
        model_name  : FastEmbed model (phải khớp với lúc ingest)
        retrieve_k  : số chunk lấy từ FAISS (trước rerank)
        top_n       : số chunk đưa vào LLM (sau rerank)
        use_rerank  : True = dùng FlashRank Rerank | False = bỏ qua
        use_cache   : True = bật Semantic Cache | False = bỏ qua
        cache       : SemanticCache instance (tái dùng nếu đã tạo)
        db          : FAISS instance (tái dùng nếu đã load)
        llm         : ChatOpenAI instance (tái dùng nếu đã khởi tạo)
        verbose     : in log ra màn hình

    Returns dict gồm:
        answer      : câu trả lời
        sources     : danh sách metadata các chunk đã dùng
        chunks      : list[Document] raw
        context     : context string đã build
        from_cache  : True nếu lấy từ cache
    """
    t_start = time.time()

    # --- Khởi tạo Cache ---
    if use_cache and cache is None:
        cache = SemanticCache(vector_dir=vector_dir, model_name=model_name)

    # --- STEP 2: Cache lookup ---
    if use_cache:
        cached = cache.lookup(question, verbose=verbose)
        if cached is not None:
            if verbose:
                elapsed = time.time() - t_start
                print(f"[CACHE] Trả về từ cache ({elapsed:.2f}s)")
                print("\n" + "=" * 56)
                print("ANSWER (from cache):")
                print(cached["answer"])
                print("=" * 56)
            return {**cached, "chunks": [], "from_cache": True}

    # --- Load FAISS nếu chưa có ---
    if db is None:
        db = load_faiss(vector_dir=vector_dir, model_name=model_name)

    # --- Khởi tạo LLM nếu chưa có ---
    if llm is None:
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=DEFAULT_TEMPERATURE)
        if verbose:
            print("[LLM] ChatOpenAI gpt-4.1-mini ready")

    # --- STEP 3: Retrieve ---
    candidates = retrieve_chunks(db, question, top_k=retrieve_k)

    # --- STEP 4: Rerank (optional) ---
    if use_rerank:
        chunks = rerank_chunks(query=question, docs=candidates, top_n=top_n)
    else:
        chunks = candidates[:top_n]
        if verbose:
            print(f"[RERANK] Skipped — lấy top {top_n} từ FAISS")

    # --- STEP 5: Build context ---
    context = build_context(chunks)
    if verbose:
        print(f"\n[CONTEXT] {len(chunks)} chunks, ~{len(context)} chars")

    # --- STEP 6-7: Generate ---
    if verbose:
        print("[LLM] Generating answer...")

    answer = generate_answer(llm, context, question)

    sources = [
        {
            "file_name": doc.metadata.get("file_name", "unknown"),
            "page":      doc.metadata.get("page", "?"),
            "chunk_id":  doc.metadata.get("semantic_chunk_id", "?"),
        }
        for doc in chunks
    ]

    result = {
        "answer":     answer,
        "sources":    sources,
        "chunks":     chunks,
        "context":    context,
        "from_cache": False,
    }

    # --- STEP 8: Lưu vào cache ---
    if use_cache:
        cache.save_entry(question, result)

    elapsed = time.time() - t_start
    if verbose:
        print("\n" + "=" * 56)
        print(f"ANSWER (generated in {elapsed:.2f}s):")
        print(answer)
        print("=" * 56)

    return result


# ======================================================
# RAGQueryEngine — class dùng trong session dài
# ======================================================

class RAGQueryEngine:
    """
    Load FAISS + LLM + Cache 1 lần → query nhiều lần.
    Cache được giữ trong memory, tự đồng bộ với file JSON.

    Ví dụ:
        engine = RAGQueryEngine()
        r1 = engine.query("RAG là gì?")              # gọi LLM, lưu cache
        r2 = engine.query("RAG hoạt động thế nào?")  # có thể hit cache
        r3 = engine.query("RAG là gì?")              # hit cache 100%
        print(r3["from_cache"])  # True

        # Bật rerank cho 1 query cụ thể:
        r4 = engine.query("câu hỏi quan trọng", use_rerank=True)

        # Xem thống kê cache:
        print(engine.cache.stats())
    """

    def __init__(
        self,
        vector_dir: str  = DEFAULT_VECTOR_DIR,
        model_name: str  = DEFAULT_MODEL,
        retrieve_k: int  = DEFAULT_RETRIEVE_K,
        top_n: int       = DEFAULT_TOP_K,
        use_rerank: bool = False,
        use_cache: bool  = True,
        cache_threshold: float = DEFAULT_CACHE_THRESHOLD,
    ):
        self.vector_dir = vector_dir
        self.model_name = model_name
        self.retrieve_k = retrieve_k
        self.top_n      = top_n
        self.use_rerank = use_rerank
        self.use_cache  = use_cache

        print("\n[RAGQueryEngine] Initializing...")
        self.db    = load_faiss(vector_dir=vector_dir, model_name=model_name)
        self.llm   = ChatOpenAI(model="gpt-4.1-mini", temperature=DEFAULT_TEMPERATURE)
        self.cache = SemanticCache(
            vector_dir=vector_dir,
            model_name=model_name,
            threshold=cache_threshold,
        ) if use_cache else None
        print("[RAGQueryEngine] Ready ✓\n")

    def query(
        self,
        question: str,
        use_rerank: Optional[bool] = None,
        use_cache: Optional[bool]  = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        return rag_query(
            question   = question,
            retrieve_k = self.retrieve_k,
            top_n      = self.top_n,
            use_rerank = use_rerank if use_rerank is not None else self.use_rerank,
            use_cache  = use_cache  if use_cache  is not None else self.use_cache,
            cache      = self.cache,
            db         = self.db,
            llm        = self.llm,
            verbose    = verbose,
        )

    def clear_cache(self):
        """Xóa toàn bộ cache."""
        if self.cache:
            self.cache.clear()


# ======================================================
# CLI ENTRY
# ======================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG Query — FAISS + Semantic Cache + FlashRank Rerank + GPT-4.1-mini"
    )
    parser.add_argument("--question",   required=True,                   help="Câu hỏi")
    parser.add_argument("--vector-dir", default=DEFAULT_VECTOR_DIR,      help="Thư mục FAISS")
    parser.add_argument("--model",      default=DEFAULT_MODEL,           help="FastEmbed model name")
    parser.add_argument("--retrieve-k", default=DEFAULT_RETRIEVE_K, type=int)
    parser.add_argument("--top-n",      default=DEFAULT_TOP_K,      type=int)
    parser.add_argument("--no-rerank",  action="store_true",             help="Tắt FlashRank Rerank")
    parser.add_argument("--no-cache",   action="store_true",             help="Tắt Semantic Cache")

    args = parser.parse_args()

    result = rag_query(
        question   = args.question,
        vector_dir = args.vector_dir,
        model_name = args.model,
        retrieve_k = args.retrieve_k,
        top_n      = args.top_n,
        use_rerank = not args.no_rerank,
        use_cache  = not args.no_cache,
    )

    print(f"\n[FROM_CACHE] {result['from_cache']}")
    print("\n[SOURCES]")
    print(json.dumps(result["sources"], indent=2, ensure_ascii=False))