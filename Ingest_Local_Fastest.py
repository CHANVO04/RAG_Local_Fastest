"""
ingest.py — Simple offline ingest pipeline for RAG (FAISS + FastEmbed)

Chức năng chính:
1. Ingest PDF mới vào FAISS
2. Chống ingest trùng bằng file hash
3. Liệt kê tài liệu đã ingest
4. Xóa tài liệu cũ khỏi registry + FAISS bằng cách rebuild lại index

Cần:
- rag_pdf_loader.py có hàm load_scientific_pdf(...)
- data/ chứa file PDF
- vector_store/ là nơi lưu FAISS, registry, config, chunk store
"""
# python ingest.py --action delete
# python ingest.py --action list
# python ingest.py --action ingest
import os
import json
import hashlib
from datetime import datetime
from typing import List, Callable, Optional, Dict, Any

import numpy as np
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pdf_loader import load_scientific_pdf as _load_pdf_v3


# ======================================================
# DEFAULT CONFIG
# ======================================================

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_SEMANTIC_THRESHOLD = 0.75
DEFAULT_SEMANTIC_MAX_CHARS = 1200

DEFAULT_CONFIG_NAME = "embedding_config.json"
DEFAULT_REGISTRY_NAME = "registry.json"
DEFAULT_DOC_STORE_NAME = "documents_store.json"


# ======================================================
# EMBEDDING — FastEmbed
# ======================================================

def get_embedding_model(model_name: str = DEFAULT_MODEL) -> FastEmbedEmbeddings:
    """Khởi tạo FastEmbedEmbeddings để dùng với LangChain FAISS."""
    print(f"[EMBED] Loading FastEmbedEmbeddings: {model_name}")
    model = FastEmbedEmbeddings(model_name=model_name)
    print("[EMBED] Model ready")
    return model


# ======================================================
# BASIC UTILITIES
# ======================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clean_scientific_text(text: str) -> str:
    """
    Cleaning rất đơn giản:
    đánh dấu Figure / Table / Equation để downstream dễ nhận diện hơn.
    """
    replace_map = {
        "Figure": "\n[FIGURE]",
        "Fig.": "\n[FIGURE]",
        "Table": "\n[TABLE]",
        "Equation": "\n[EQUATION]",
    }
    for old, new in replace_map.items():
        text = text.replace(old, new)
    return text


def calculate_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """
    Tính SHA256 theo binary content của file.
    Dùng để check file đã ingest chưa.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


# ======================================================
# PATH HELPERS
# ======================================================

def get_registry_path(vector_dir: str) -> str:
    return os.path.join(vector_dir, DEFAULT_REGISTRY_NAME)


def get_doc_store_path(vector_dir: str) -> str:
    return os.path.join(vector_dir, DEFAULT_DOC_STORE_NAME)


def get_config_path(vector_dir: str) -> str:
    return os.path.join(vector_dir, DEFAULT_CONFIG_NAME)


# ======================================================
# REGISTRY
# ======================================================

def load_registry(vector_dir: str) -> Dict[str, Any]:
    """
    registry.json lưu danh sách file đã ingest.
    """
    path = get_registry_path(vector_dir)
    if not os.path.exists(path):
        return {"documents": []}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry: Dict[str, Any], vector_dir: str):
    ensure_dir(vector_dir)
    path = get_registry_path(vector_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    print(f"[REGISTRY] Saved → {path}")


def is_file_already_ingested(file_hash: str, registry: Dict[str, Any]) -> bool:
    """
    Nếu file hash đã có trong registry thì coi như đã ingest rồi.
    """
    for item in registry.get("documents", []):
        if item.get("file_hash") == file_hash:
            return True
    return False


def add_file_to_registry(
    registry: Dict[str, Any],
    file_name: str,
    file_hash: str,
    file_size: int,
    total_pages: int,
    total_raw_chunks: int,
    total_semantic_chunks: int,
):
    registry.setdefault("documents", []).append({
        "file_name": file_name,
        "file_hash": file_hash,
        "file_size": file_size,
        "total_pages": total_pages,
        "total_raw_chunks": total_raw_chunks,
        "total_semantic_chunks": total_semantic_chunks,
        "ingested_at": datetime.now().isoformat(),
    })


def list_ingested_documents(vector_dir: str) -> List[Dict[str, Any]]:
    """
    Liệt kê tất cả tài liệu đã ingest từ registry.json.
    """
    registry = load_registry(vector_dir)
    return registry.get("documents", [])


# ======================================================
# DOCUMENT STORE
# ======================================================

def load_documents_store(vector_dir: str) -> Dict[str, Any]:
    """
    documents_store.json lưu semantic chunks dạng text + metadata.
    File này dùng để rebuild lại FAISS khi xóa tài liệu.
    """
    path = get_doc_store_path(vector_dir)
    if not os.path.exists(path):
        return {"documents": []}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_documents_store(store: Dict[str, Any], vector_dir: str):
    ensure_dir(vector_dir)
    path = get_doc_store_path(vector_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)
    print(f"[DOC_STORE] Saved → {path}")


def append_semantic_chunks_to_store(semantic_chunks: List[Document], vector_dir: str):
    """
    Lưu semantic chunks vào documents_store.json
    để sau này có thể rebuild FAISS.
    """
    store = load_documents_store(vector_dir)

    for doc in semantic_chunks:
        store["documents"].append({
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        })

    save_documents_store(store, vector_dir)


def load_documents_from_store(vector_dir: str) -> List[Document]:
    """
    Đọc semantic chunks từ documents_store.json
    và convert lại thành list[Document].
    """
    store = load_documents_store(vector_dir)
    docs = []

    for item in store.get("documents", []):
        docs.append(
            Document(
                page_content=item["page_content"],
                metadata=item["metadata"],
            )
        )

    return docs


# ======================================================
# STEP 1 — FIND PDF FILES
# ======================================================

def find_pdf_files(data_dir: str) -> List[str]:
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    return sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    ])


# ======================================================
# STEP 2 — LOAD ONLY NEW PDF FILES
# ======================================================

def load_new_pdfs(
    pdf_paths: List[str],
    registry: Dict[str, Any],
    loader_fn: Callable[[str, bool], Any] = _load_pdf_v3,
    verbose: bool = True,
) -> tuple[List[Document], List[Dict[str, Any]], List[str]]:
    """
    Chỉ load các file chưa từng ingest.
    """
    documents: List[Document] = []
    new_files_info: List[Dict[str, Any]] = []
    skipped_files: List[str] = []

    for pdf_path in pdf_paths:
        file_name = os.path.basename(pdf_path)
        file_hash = calculate_file_hash(pdf_path)
        file_size = os.path.getsize(pdf_path)

        if is_file_already_ingested(file_hash, registry):
            if verbose:
                print(f"[SKIP] {file_name} đã tồn tại trong hệ thống")
            skipped_files.append(file_name)
            continue

        if verbose:
            print(f"📄 Loading new file: {file_name}")

        result = loader_fn(pdf_path, verbose=verbose)
        pages = result.to_langchain_documents()

        for page in pages:
            page.metadata["document_id"] = file_name
            page.metadata["file_name"] = file_name
            page.metadata["file_hash"] = file_hash
            page.page_content = clean_scientific_text(page.page_content)

        documents.extend(pages)

        new_files_info.append({
            "file_name": file_name,
            "file_hash": file_hash,
            "file_size": file_size,
            "total_pages": len(pages),
        })

        if verbose:
            print(f"   → loaded {len(pages)} page-documents")

    return documents, new_files_info, skipped_files


# ======================================================
# STEP 3 — CHUNKING
# ======================================================

def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    avg_chars = sum(len(c.page_content) for c in chunks) // len(chunks)
    print(f"✂️  Created {len(chunks)} chunks (avg {avg_chars} chars/chunk)")
    return chunks


# ======================================================
# STEP 4 — SEMANTIC MERGE
# ======================================================

def semantic_merge_chunks(
    chunks: List[Document],
    embedding_model: Optional[FastEmbedEmbeddings] = None,
    threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    max_chars: int = DEFAULT_SEMANTIC_MAX_CHARS,
) -> List[Document]:
    """
    Merge các chunk liền kề nếu:
    - cùng document_id
    - similarity đủ cao
    - độ dài sau merge không vượt max_chars

    Dùng embed_documents() / embed_query() của LangChain thay cho
    SentenceTransformer.encode(). Kết quả được ép kiểu sang np.array
    trước khi đưa vào np.dot() để tránh lỗi list multiplication.
    """
    if not chunks:
        return []

    if embedding_model is None:
        embedding_model = get_embedding_model()

    print("[SEMANTIC] Running semantic merge...")

    texts = [c.page_content for c in chunks]

    # embed_documents() trả về List[List[float]] → ép sang np.array
    embeddings = np.array(embedding_model.embed_documents(texts))

    merged: List[Document] = []

    current_text = chunks[0].page_content
    current_meta = dict(chunks[0].metadata)
    current_raw_ids = [chunks[0].metadata.get("chunk_id")]
    current_emb = embeddings[0]
    semantic_id = 0

    for i in range(1, len(chunks)):
        next_chunk = chunks[i]
        next_emb = embeddings[i]

        same_doc = current_meta.get("document_id") == next_chunk.metadata.get("document_id")
        sim = float(np.dot(current_emb, next_emb))
        candidate_text = current_text + "\n\n" + next_chunk.page_content

        if same_doc and sim >= threshold and len(candidate_text) <= max_chars:
            current_text = candidate_text
            current_raw_ids.append(next_chunk.metadata.get("chunk_id"))
            # embed_query() trả về List[float] → ép sang np.array
            current_emb = np.array(embedding_model.embed_query(current_text))
        else:
            current_meta["semantic_chunk_id"] = semantic_id
            current_meta["raw_chunk_ids"] = current_raw_ids.copy()

            merged.append(
                Document(
                    page_content=current_text,
                    metadata=current_meta.copy(),
                )
            )

            semantic_id += 1
            current_text = next_chunk.page_content
            current_meta = dict(next_chunk.metadata)
            current_raw_ids = [next_chunk.metadata.get("chunk_id")]
            current_emb = next_emb

    current_meta["semantic_chunk_id"] = semantic_id
    current_meta["raw_chunk_ids"] = current_raw_ids.copy()
    merged.append(
        Document(
            page_content=current_text,
            metadata=current_meta.copy(),
        )
    )

    print(f"[SEMANTIC] Raw chunks: {len(chunks)} → Semantic chunks: {len(merged)}")
    return merged


# ======================================================
# STEP 5 — BUILD / UPDATE / REBUILD FAISS
# ======================================================

def build_or_update_faiss(
    semantic_chunks: List[Document],
    vector_dir: str,
    embedding_model: Optional[FastEmbedEmbeddings] = None,
) -> Optional[FAISS]:
    """
    Nếu đã có FAISS thì add thêm chunks mới.
    Nếu chưa có thì tạo mới.
    """
    ensure_dir(vector_dir)

    if embedding_model is None:
        embedding_model = get_embedding_model()

    index_file = os.path.join(vector_dir, "index.faiss")
    store_file = os.path.join(vector_dir, "index.pkl")

    if not semantic_chunks:
        if os.path.exists(index_file) and os.path.exists(store_file):
            print("[DB] No new chunks. Loading existing FAISS...")
            return FAISS.load_local(
                folder_path=vector_dir,
                embeddings=embedding_model,
                allow_dangerous_deserialization=True,
            )
        return None

    if os.path.exists(index_file) and os.path.exists(store_file):
        print("[DB] Loading existing FAISS...")
        db = FAISS.load_local(
            folder_path=vector_dir,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True,
        )
        print(f"[DB] Adding {len(semantic_chunks)} new semantic chunks...")
        db.add_documents(semantic_chunks)
    else:
        print("[DB] Creating new FAISS index...")
        db = FAISS.from_documents(semantic_chunks, embedding_model)

    db.save_local(vector_dir)
    print(f"[DB] FAISS saved → {vector_dir}")
    return db


def rebuild_faiss_from_store(
    vector_dir: str,
    model_name: str = DEFAULT_MODEL,
) -> Optional[FAISS]:
    """
    Rebuild lại toàn bộ FAISS từ documents_store.json.
    Đây là cách đơn giản nhất để hỗ trợ xóa document trong FAISS local.
    """
    docs = load_documents_from_store(vector_dir)

    index_file = os.path.join(vector_dir, "index.faiss")
    store_file = os.path.join(vector_dir, "index.pkl")

    # Nếu không còn document nào thì xóa hẳn index cũ
    if not docs:
        if os.path.exists(index_file):
            os.remove(index_file)
        if os.path.exists(store_file):
            os.remove(store_file)

        print("[DB] No documents left. FAISS index removed.")
        return None

    embedding_model = get_embedding_model(model_name=model_name)

    print(f"[DB] Rebuilding FAISS from {len(docs)} stored semantic chunks...")
    db = FAISS.from_documents(docs, embedding_model)
    db.save_local(vector_dir)
    print(f"[DB] Rebuilt FAISS saved → {vector_dir}")
    return db


# ======================================================
# CONFIG
# ======================================================

def save_config(config: Dict[str, Any], vector_dir: str, filename: str = DEFAULT_CONFIG_NAME) -> str:
    ensure_dir(vector_dir)
    path = os.path.join(vector_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[CONFIG] Saved → {path}")
    return path


# ======================================================
# HELPER STATS
# ======================================================

def count_chunks_per_file(chunks: List[Document], key: str = "file_name") -> Dict[str, int]:
    """
    Đếm số chunk theo từng file_name.
    """
    stats: Dict[str, int] = {}
    for chunk in chunks:
        name = chunk.metadata.get(key, "unknown")
        stats[name] = stats.get(name, 0) + 1
    return stats


# ======================================================
# MAIN INGEST PIPELINE
# ======================================================

def offline_ingest(
    data_dir: str = "data",
    vector_dir: str = "vector_store",
    model_name: str = DEFAULT_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    semantic_max_chars: int = DEFAULT_SEMANTIC_MAX_CHARS,
    loader_fn: Callable[[str, bool], Any] = _load_pdf_v3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Luồng ingest:
    1. Tìm PDF
    2. Đọc registry
    3. Bỏ qua file đã ingest
    4. Load file mới
    5. Chunk
    6. Semantic merge
    7. Update FAISS
    8. Update registry
    9. Update documents_store
    10. Save config
    """
    if verbose:
        print("\n" + "=" * 56)
        print("  OFFLINE INGEST PIPELINE")
        print(f"  model      : {model_name}")
        print(f"  vector_dir : {vector_dir}")
        print("=" * 56)

    ensure_dir(vector_dir)

    pdf_paths = find_pdf_files(data_dir)
    if verbose:
        print(f"[DATA] Found {len(pdf_paths)} PDF file(s)")

    registry = load_registry(vector_dir)

    documents, new_files_info, skipped_files = load_new_pdfs(
        pdf_paths=pdf_paths,
        registry=registry,
        loader_fn=loader_fn,
        verbose=verbose,
    )

    if not documents:
        if verbose:
            print("[INFO] Không có file mới để ingest.")

        config = {
            "mode": "offline",
            "model": model_name,
            "vector_db": "faiss",
            "pdf_files_found": [os.path.basename(p) for p in pdf_paths],
            "new_pdf_files": [],
            "skipped_pdf_files": skipped_files,
            "total_documents": 0,
            "total_raw_chunks": 0,
            "total_semantic_chunks": 0,
        }
        config_path = save_config(config, vector_dir)

        return {
            "status": "no_new_files",
            "new_pdf_files": [],
            "skipped_pdf_files": skipped_files,
            "total_documents": 0,
            "total_raw_chunks": 0,
            "total_semantic_chunks": 0,
            "vector_dir": vector_dir,
            "config_file": config_path,
            "faiss": None,
        }

    chunks = chunk_documents(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Khởi tạo 1 lần duy nhất — dùng cho cả semantic merge lẫn FAISS
    embedding_model = get_embedding_model(model_name=model_name)

    semantic_chunks = semantic_merge_chunks(
        chunks=chunks,
        embedding_model=embedding_model,
        threshold=semantic_threshold,
        max_chars=semantic_max_chars,
    )

    db = build_or_update_faiss(
        semantic_chunks=semantic_chunks,
        vector_dir=vector_dir,
        embedding_model=embedding_model,
    )

    raw_chunk_count_by_file = count_chunks_per_file(chunks, key="file_name")
    semantic_chunk_count_by_file = count_chunks_per_file(semantic_chunks, key="file_name")

    for info in new_files_info:
        file_name = info["file_name"]

        add_file_to_registry(
            registry=registry,
            file_name=file_name,
            file_hash=info["file_hash"],
            file_size=info["file_size"],
            total_pages=info["total_pages"],
            total_raw_chunks=raw_chunk_count_by_file.get(file_name, 0),
            total_semantic_chunks=semantic_chunk_count_by_file.get(file_name, 0),
        )

    save_registry(registry, vector_dir)
    append_semantic_chunks_to_store(semantic_chunks, vector_dir)

    config = {
        "mode": "offline",
        "model": model_name,
        "vector_db": "faiss",
        "loader": getattr(loader_fn, "__name__", "rag_pdf_loader_v3"),
        "pdf_files_found": [os.path.basename(p) for p in pdf_paths],
        "new_pdf_files": [x["file_name"] for x in new_files_info],
        "skipped_pdf_files": skipped_files,
        "total_documents": len(documents),
        "total_raw_chunks": len(chunks),
        "total_semantic_chunks": len(semantic_chunks),
        "semantic_threshold": semantic_threshold,
        "semantic_max_chars": semantic_max_chars,
    }
    config_path = save_config(config, vector_dir)

    return {
        "status": "success",
        "new_pdf_files": config["new_pdf_files"],
        "skipped_pdf_files": skipped_files,
        "total_documents": len(documents),
        "total_raw_chunks": len(chunks),
        "total_semantic_chunks": len(semantic_chunks),
        "vector_dir": vector_dir,
        "config_file": config_path,
        "faiss": db,
    }


# ======================================================
# DELETE DOCUMENT
# ======================================================

def delete_document(
    file_name: str,
    vector_dir: str = "vector_store",
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Xóa 1 tài liệu theo file_name.

    Cách làm:
    1. Xóa document đó khỏi registry.json
    2. Xóa toàn bộ semantic chunks của nó khỏi documents_store.json
    3. Rebuild lại FAISS từ phần còn lại
    """
    registry = load_registry(vector_dir)
    store = load_documents_store(vector_dir)

    old_registry_docs = registry.get("documents", [])
    old_store_docs = store.get("documents", [])

    # Lọc registry
    new_registry_docs = [
        item for item in old_registry_docs
        if item.get("file_name") != file_name
    ]

    # Lọc documents_store
    new_store_docs = [
        item for item in old_store_docs
        if item.get("metadata", {}).get("file_name") != file_name
    ]

    removed_registry_count = len(old_registry_docs) - len(new_registry_docs)
    removed_chunk_count = len(old_store_docs) - len(new_store_docs)

    # Nếu không xóa được gì thì báo luôn
    if removed_registry_count == 0 and removed_chunk_count == 0:
        return {
            "status": "not_found",
            "file_name": file_name,
            "message": f"Không tìm thấy tài liệu '{file_name}' trong hệ thống.",
        }

    registry["documents"] = new_registry_docs
    store["documents"] = new_store_docs

    save_registry(registry, vector_dir)
    save_documents_store(store, vector_dir)

    # Rebuild lại FAISS
    db = rebuild_faiss_from_store(vector_dir, model_name=model_name)

    return {
        "status": "deleted",
        "file_name": file_name,
        "removed_registry_count": removed_registry_count,
        "removed_chunk_count": removed_chunk_count,
        "faiss": db,
        "message": f"Đã xóa tài liệu '{file_name}' và rebuild lại FAISS.",
    }


# ======================================================
# CLI ENTRY
# ======================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple offline ingest (FAISS + FastEmbed)")
    parser.add_argument("--data", default="data", help="Folder chứa PDF")
    parser.add_argument("--vector-dir", default="vector_store", help="Folder lưu FAISS")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="FastEmbed model name")

    # action
    parser.add_argument("--action", default="ingest", choices=["ingest", "list", "delete"], help="Hành động muốn chạy")
    parser.add_argument("--file-name", default="", help="Tên file cần xóa khi dùng --action delete")

    args = parser.parse_args()

    if args.action == "ingest":
        result = offline_ingest(
            data_dir=args.data,
            vector_dir=args.vector_dir,
            model_name=args.model,
        )
        print(json.dumps({k: v for k, v in result.items() if k != "faiss"}, indent=2, ensure_ascii=False))

    elif args.action == "list":
        docs = list_ingested_documents(args.vector_dir)
        print(json.dumps(docs, indent=2, ensure_ascii=False))

    elif args.action == "delete":
        if not args.file_name:
            raise ValueError("Bạn phải truyền --file-name khi dùng --action delete")

        result = delete_document(
            file_name=args.file_name,
            vector_dir=args.vector_dir,
            model_name=args.model,
        )
        print(json.dumps({k: v for k, v in result.items() if k != "faiss"}, indent=2, ensure_ascii=False))