"""
========================================================
rag_pdf_loader.py  —  RAG PDF Loader Library  v3.0
========================================================

Sử dụng:
    from rag_pdf_loader import load_scientific_pdf, RagPdfResult

    result = load_scientific_pdf("paper.pdf")

    # Truy cập kết quả
    result.pages           # List[PageDict]
    result.layout          # List[BlockDict]
    result.column_layout   # Dict[int, str]  — "double" | "single"
    result.tables          # Dict[int, List[List]]
    result.metadata        # MetadataDict
    result.sections        # List[SectionDict]

    # Lấy danh sách LangChain Document (dùng cho chunking)
    docs = result.to_langchain_documents()

Yêu cầu:
    pip install pymupdf camelot-py[cv] langchain-community

Đặc điểm
--------
    - CPU-friendly, không OCR, không deep-learning
    - Tối ưu cho academic paper (research + survey)
    - Phát hiện layout 2 cột tự động
    - Lọc false-positive table (author block / citation / text paragraph)
    - Section detection với 50+ patterns (research + survey paper)
    - Metadata fallback: author từ layout, arXiv ID

Changelog
---------
    v3.0  : Đóng gói thành thư viện, thêm RagPdfResult, to_langchain_documents()
    v3.0  : Step 4 — thêm "Table N" text signal, ruling-lines threshold=3
    v3.0  : Step 6 — blacklist institutional keywords cho author fallback
    v3.0  : Step 7 — regex exact/numbered match, bỏ isolation rule
    v2.0  : Step 2 — garbage detection
    v2.0  : Step 3 — column label + analyze_column_layout
    v2.0  : Step 5 — 3-tầng post-filter
    v2.0  : Step 6 — arXiv ID extraction
"""

# ======================================================
# IMPORTS
# ======================================================

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import fitz
import camelot
from langchain_community.document_loaders import PyMuPDFLoader
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


# ======================================================
# PUBLIC TYPES
# ======================================================

PageDict    = Dict[str, Any]   # {page, raw_text, text, corrupted}
BlockDict   = Dict[str, Any]   # {page, bbox, text, column, corrupted}
SectionDict = Dict[str, Any]   # {section, text, is_heading}
MetadataDict = Dict[str, Any]  # {title, author, producer, ...}


@dataclass
class RagPdfResult:
    """
    Kết quả trả về từ load_scientific_pdf().
    Tất cả các trường đều có thể truy cập trực tiếp.
    """
    pages:         List[PageDict]         = field(default_factory=list)
    layout:        List[BlockDict]        = field(default_factory=list)
    column_layout: Dict[int, str]         = field(default_factory=dict)
    tables:        Dict[int, List]        = field(default_factory=dict)
    metadata:      MetadataDict           = field(default_factory=dict)
    sections:      List[SectionDict]      = field(default_factory=list)

    def to_langchain_documents(self) -> List[Document]:
        """
        Chuyển đổi sang List[Document] của LangChain.

        Mỗi Document tương ứng với một trang PDF:
            - page_content : nội dung văn bản đã clean
            - metadata     : title, author, doi, arxiv_id, page, total_pages,
                             section_label (section chiếm nhiều dòng nhất trên trang)
                             tables_on_page, corrupted

        Trang bị đánh dấu corrupted sẽ bị bỏ qua.
        """
        # Tạo map: page_number → section_label chiếm nhiều nhất
        page_section_map = self._build_page_section_map()

        # Tạo map: page_number → danh sách tables
        page_table_map: Dict[int, List] = {}
        for pg, tlist in self.tables.items():
            page_table_map[pg] = tlist

        docs = []
        for p in self.pages:
            if p.get("corrupted"):
                continue
            text = p.get("text", "").strip()
            if not text:
                continue

            pg_num = p["page"]
            doc = Document(
                page_content=text,
                metadata={
                    # Nguồn gốc
                    "page":           pg_num,
                    "total_pages":    self.metadata.get("total_pages"),
                    # Nhận dạng tài liệu
                    "title":          self.metadata.get("title", ""),
                    "author":         self.metadata.get("author", ""),
                    "doi":            self.metadata.get("doi"),
                    "arxiv_id":       self.metadata.get("arxiv_id"),
                    "producer":       self.metadata.get("producer", ""),
                    "creationDate":   self.metadata.get("creationDate", ""),
                    # Ngữ cảnh nội dung
                    "section_label":  page_section_map.get(pg_num, "unknown"),
                    "column_layout":  self.column_layout.get(pg_num, "unknown"),
                    "tables_on_page": page_table_map.get(pg_num, []),
                    "corrupted":      False,
                }
            )
            docs.append(doc)
        return docs

    def _build_page_section_map(self) -> Dict[int, str]:
        """
        Với mỗi trang, lấy section_label xuất hiện nhiều lần nhất
        (dựa trên các dòng section trong full_text).

        Vì sections là list theo dòng và không có page info trực tiếp,
        ta estimate bằng cách split full_text theo trang.
        """
        # Ghép index trang vào sections dựa trên page text boundaries
        page_sections: Dict[int, Dict[str, int]] = {}
        section_cursor = "unknown"
        section_idx = 0

        total_lines = len(self.sections)
        if total_lines == 0:
            return {}

        # Tính số dòng xấp xỉ mỗi trang (sections tương ứng với full_text đã ghép)
        lines_per_page_approx: Dict[int, int] = {}
        for p in self.pages:
            if not p.get("corrupted"):
                lines_per_page_approx[p["page"]] = len(p.get("text", "").split("\n"))

        # Phân bổ sections theo trang
        current_page_idx = 0
        non_corrupted_pages = [p["page"] for p in self.pages if not p.get("corrupted")]
        line_counter = 0

        for pg in non_corrupted_pages:
            pg_line_count = lines_per_page_approx.get(pg, 0)
            page_sections[pg] = {}

            for _ in range(pg_line_count):
                if section_idx >= total_lines:
                    break
                sec = self.sections[section_idx]["section"]
                page_sections[pg][sec] = page_sections[pg].get(sec, 0) + 1
                section_idx += 1

        # Lấy section chiếm nhiều nhất mỗi trang
        result = {}
        for pg, counts in page_sections.items():
            if counts:
                result[pg] = max(counts, key=counts.get)
        return result

    def summary(self) -> str:
        """Trả về chuỗi tóm tắt kết quả pipeline."""
        corrupted = sum(1 for p in self.pages if p.get("corrupted"))
        col_double = sum(1 for v in self.column_layout.values() if v == "double")
        section_counts: Dict[str, int] = {}
        for s in self.sections:
            sec = s["section"]
            section_counts[sec] = section_counts.get(sec, 0) + 1
        top = sorted(section_counts.items(), key=lambda x: -x[1])[:6]
        detected = sorted(set(s["section"] for s in self.sections if s["section"] != "unknown"))

        lines = [
            "─" * 56,
            "  RAG PDF Loader v3.0 — Summary",
            "─" * 56,
            f"  Title          : {self.metadata.get('title','')[:55]}",
            f"  Author         : {str(self.metadata.get('author',''))[:55]}",
            f"  ID             : {self.metadata.get('doi') or self.metadata.get('arxiv_id') or 'N/A'}",
            f"  Pages          : {len(self.pages)}  (corrupted: {corrupted})",
            f"  Layout blocks  : {len(self.layout)}",
            f"  Column layout  : {col_double} double / {len(self.column_layout) - col_double} single",
            f"  Tables found   : {sum(len(v) for v in self.tables.values())}",
            f"  Sections       : {detected}",
            f"  Top labels     : {dict(top)}",
            "─" * 56,
        ]
        return "\n".join(lines)


# ======================================================
# INTERNAL CONFIGURATION
# ======================================================

_RESEARCH_PATTERNS = [
    "abstract", "introduction", "related work", "background",
    "preliminary", "preliminaries", "problem formulation",
    "problem statement", "method", "methods", "methodology",
    "approach", "proposed method", "model", "framework",
    "architecture", "experiment", "experiments",
    "experimental setup", "experimental results", "evaluation",
    "results", "analysis", "ablation", "discussion",
    "conclusion", "conclusions", "future work", "limitations",
    "acknowledgment", "acknowledgments", "acknowledgements",
    "references", "appendix", "supplementary",
]

_SURVEY_PATTERNS = [
    "overview", "taxonomy", "survey", "review", "summary",
    "categorization", "classification", "comparison", "benchmark",
    "dataset", "datasets", "task", "tasks", "application",
    "applications", "challenge", "challenges", "opportunity",
    "opportunities", "trend", "trends", "future", "practical",
    "implementation", "optimization", "downstream", "cost",
    "risk", "pattern", "patterns", "purpose", "definition",
    "motivation", "contribution", "contributions", "guide",
]

_ALL_SECTION_PATTERNS: List[str] = list(dict.fromkeys(
    _RESEARCH_PATTERNS + _SURVEY_PATTERNS
))

_GARBAGE_RATIO_THRESHOLD = 0.25
_TABLE_MIN_ACCURACY      = 70
_TABLE_MAX_WHITESPACE    = 90
_TABLE_MAX_CITE_RATIO    = 0.5
_TABLE_MAX_CELL_LEN      = 80

_INSTITUTIONAL_KW = re.compile(
    r"\b(university|institute|laboratory|lab|department|dept|"
    r"college|school|center|centre|research|technology|science|"
    r"engineering|national|international|china|usa|france|"
    r"germany|japan|korea|uk|canada|beijing|shanghai|"
    r"computing|intelligence|autonomous|systems|data|ai)\b",
    re.I
)

_SECTION_ALIASES = {
    "methods": "method", "experiments": "experiment",
    "conclusions": "conclusion", "acknowledgments": "acknowledgment",
    "acknowledgements": "acknowledgment", "preliminaries": "preliminary",
    "contributions": "contribution", "applications": "application",
    "challenges": "challenge", "datasets": "dataset",
    "tasks": "task", "trends": "trend",
    "patterns": "pattern", "opportunities": "opportunity",
}


def _build_section_regex() -> re.Pattern:
    escaped = [re.escape(s) for s in sorted(_ALL_SECTION_PATTERNS, key=len, reverse=True)]
    joined  = "|".join(escaped)
    return re.compile(
        rf"^(?:\d+(?:\.\d+)*\.?\s+)?({joined})[\s:]*$", re.I
    )


_SECTION_RE = _build_section_regex()


# ======================================================
# STEP 2 — TEXT CLEANING (internal)
# ======================================================

def _is_garbage(text: str) -> bool:
    if not text or len(text) < 10:
        return False
    bad = 0
    for ch in text:
        cp = ord(ch)
        if 32 <= cp <= 126 or cp in (9, 10, 13):
            continue
        if (0x00C0 <= cp <= 0x024F or 0x0370 <= cp <= 0x03FF
                or 0x2000 <= cp <= 0x206F or 0x2100 <= cp <= 0x214F
                or 0x2190 <= cp <= 0x21FF or 0x2200 <= cp <= 0x22FF
                or 0x2600 <= cp <= 0x26FF or 0x4E00 <= cp <= 0x9FFF
                or 0xAC00 <= cp <= 0xD7AF or 0x0400 <= cp <= 0x04FF):
            continue
        bad += 1
    return bad / len(text) > _GARBAGE_RATIO_THRESHOLD


def _clean(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ======================================================
# STEP 1 — PDF PARSING (internal)
# ======================================================

def _parse_pages(pdf_path: Path) -> List[PageDict]:
    t0     = time.time()
    loader = PyMuPDFLoader(str(pdf_path))
    docs   = loader.load()
    pages  = []
    for i, d in enumerate(docs):
        raw       = d.page_content or ""
        corrupted = _is_garbage(raw)
        pages.append({
            "page":      i + 1,
            "raw_text":  raw,
            "text":      _clean(raw) if not corrupted else "",
            "corrupted": corrupted,
        })
    n_bad = sum(1 for p in pages if p["corrupted"])
    print(f"[STEP 1] Parsed {len(pages)} pages in {time.time()-t0:.2f}s  "
          f"| corrupted: {n_bad}")
    return pages


# ======================================================
# STEP 3 — LAYOUT DETECTION (internal)
# ======================================================

def _col_label(x0: float, x1: float, page_w: float) -> str:
    mid = page_w / 2
    if (x1 - x0) > page_w * 0.6:
        return "full"
    if x1 < mid + 20:
        return "left"
    if x0 > mid - 20:
        return "right"
    return "full"


def _detect_layout(pdf_path: Path) -> List[BlockDict]:
    t0  = time.time()
    doc = fitz.open(pdf_path)
    out = []
    for page in doc:
        pw = page.rect.width
        for b in page.get_text("blocks"):
            x0, y0, x1, y1, txt = b[:5]
            if len(b) > 6 and b[6] != 0:
                continue
            cleaned = _clean(txt)
            if not cleaned:
                continue
            out.append({
                "page":      page.number + 1,
                "bbox":      (x0, y0, x1, y1),
                "text":      cleaned,
                "column":    _col_label(x0, x1, pw),
                "corrupted": _is_garbage(txt),
            })
    print(f"[STEP 3] Layout: {len(out)} blocks in {time.time()-t0:.2f}s")
    return out


def _analyze_columns(blocks: List[BlockDict]) -> Dict[int, str]:
    pages: Dict[int, List] = {}
    for b in blocks:
        pages.setdefault(b["page"], []).append(b)
    out = {}
    for pg, blks in pages.items():
        total   = len(blks)
        two_col = sum(1 for b in blks if b["column"] in ("left", "right"))
        out[pg] = "double" if total and two_col / total > 0.5 else "single"
    n_double = sum(1 for v in out.values() if v == "double")
    print(f"[STEP 3b] Columns: {n_double}/{len(out)} double-col pages")
    return out


# ======================================================
# STEP 4 — TABLE CANDIDATE DETECTION (internal)
# ======================================================

def _has_ruling_lines(page: fitz.Page, thr: int = 3) -> bool:
    try:
        lines = sum(
            1 for d in page.get_drawings()
            for item in d.get("items", [])
            if isinstance(item, list) and item[0] == "l"
        )
        return lines >= thr
    except Exception:
        return False


def _real_num_count(text: str) -> int:
    no_cite = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)
    return len(re.findall(r"\b\d+[.,\d]*[%]?\b", no_cite))


def _has_table_label(page: fitz.Page) -> bool:
    return bool(re.search(r"\bTable\s+\d+", page.get_text("text"), re.I))


def _find_table_candidates(pdf_path: Path,
                            col_layout: Optional[Dict] = None) -> List[int]:
    t0   = time.time()
    doc  = fitz.open(pdf_path)
    cands = []
    for page in doc:
        pn   = page.number + 1
        text = page.get_text("text")
        if len(text.strip()) < 10:
            continue
        if _has_ruling_lines(page, thr=3):
            cands.append(pn)
            continue
        if _has_table_label(page):
            cands.append(pn)
            continue
        nb = sum(1 for b in page.get_text("blocks") if _real_num_count(b[4]) >= 3)
        if nb >= 3:
            cands.append(pn)
    print(f"[STEP 4] Candidates: {cands}  in {time.time()-t0:.2f}s")
    return cands


# ======================================================
# STEP 5 — TABLE EXTRACTION (internal)
# ======================================================

_EMAIL_RE  = re.compile(r"@\w+\.\w+")
_AFFIL_RE  = re.compile(
    r"\b(university|institute|laboratory|department|college|"
    r"china|usa|france|germany|japan|korea|uk|canada)\b", re.I)
_CITE_RE   = re.compile(r"^\s*\[[\d,\s]+\]\s*$")


def _is_author_table(rows: List[List]) -> bool:
    flat = [str(c) for r in rows for c in r]
    if not flat:
        return False
    return (_EMAIL_RE.search(" ".join(flat)) is not None
            or sum(1 for c in flat if _AFFIL_RE.search(c)) >= len(flat) * 0.3)


def _is_cite_table(rows: List[List]) -> bool:
    flat = [str(c) for r in rows for c in r if str(c).strip()]
    return bool(flat) and sum(1 for c in flat if _CITE_RE.match(c)) / len(flat) > _TABLE_MAX_CITE_RATIO


def _is_text_table(rows: List[List]) -> bool:
    flat = [str(c) for r in rows for c in r if str(c).strip()]
    return bool(flat) and sum(1 for c in flat if len(c) > _TABLE_MAX_CELL_LEN) / len(flat) > 0.4


def _extract_tables(pdf_path: Path,
                    candidates: List[int]) -> Dict[int, List]:
    t0    = time.time()
    doc   = fitz.open(pdf_path)
    out   = {}
    stats = {"total": 0, "passed": 0,
             "q": 0, "auth": 0, "cite": 0, "para": 0}

    for p in candidates:
        flavor = "lattice" if _has_ruling_lines(doc.load_page(p - 1), thr=3) else "stream"
        try:
            raw = camelot.read_pdf(str(pdf_path), pages=str(p), flavor=flavor)
        except Exception:
            continue
        valid = []
        for t in raw:
            stats["total"] += 1
            rep = t.parsing_report
            if (rep.get("accuracy", 0) < _TABLE_MIN_ACCURACY
                    or rep.get("whitespace", 100) > _TABLE_MAX_WHITESPACE):
                stats["q"] += 1
                continue
            df = t.df
            if df.shape[0] <= 1 or df.shape[1] <= 1:
                stats["q"] += 1
                continue
            rows = df.values.tolist()
            if _is_author_table(rows):
                stats["auth"] += 1
                continue
            if _is_cite_table(rows):
                stats["cite"] += 1
                continue
            if _is_text_table(rows):
                stats["para"] += 1
                continue
            stats["passed"] += 1
            valid.append(rows)
        if valid:
            out[p] = valid

    print(f"[STEP 5] Tables: scanned={stats['total']} passed={stats['passed']} "
          f"rejected(q={stats['q']} auth={stats['auth']} "
          f"cite={stats['cite']} para={stats['para']})  "
          f"in {time.time()-t0:.2f}s")
    return out


# ======================================================
# STEP 6 — METADATA EXTRACTION (internal)
# ======================================================

def _fallback_author(blocks: List[BlockDict]) -> str:
    NAME_RE = re.compile(
        r"\b([A-Z][a-z]+(?:[-\s][A-Z][a-z]+)+|[A-Z]\.\s*[A-Z][a-z]+)\b"
    )
    pg1 = sorted(
        [b for b in blocks if b["page"] == 1
         and not b.get("corrupted") and b["bbox"][1] < 320],
        key=lambda b: b["bbox"][1]
    )
    authors = []
    for b in pg1[1:12]:
        for m in NAME_RE.findall(b["text"]):
            m = m.strip()
            if 5 <= len(m) <= 40 and not _INSTITUTIONAL_KW.search(m):
                authors.append(m)
    return "; ".join(dict.fromkeys(authors))


def _extract_metadata(pdf_path: Path,
                      blocks: Optional[List[BlockDict]] = None) -> MetadataDict:
    t0  = time.time()
    doc = fitz.open(pdf_path)
    meta = doc.metadata or {}

    fp_text = doc.load_page(0).get_text() if len(doc) > 0 else ""

    doi_m = re.search(r"\b10\.\d{4,9}/[^\s,;>\"')]+", fp_text)
    doi   = doi_m.group(0).rstrip(".,)") if doi_m else None

    arxiv = None
    if not doi:
        ax = re.search(r"arXiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?)", fp_text, re.I)
        if ax:
            arxiv = ax.group(1)

    author = meta.get("author", "").strip()
    if not author and blocks:
        author = _fallback_author(blocks)

    result: MetadataDict = {
        "title":        meta.get("title", "").strip(),
        "author":       author,
        "producer":     meta.get("producer", "").strip(),
        "creationDate": meta.get("creationDate", ""),
        "doi":          doi,
        "arxiv_id":     arxiv,
        "total_pages":  len(doc),
    }
    print(f"[STEP 6] Metadata in {time.time()-t0:.2f}s  "
          f"| title='{result['title'][:40]}' "
          f"doi={result['doi']} arxiv={result['arxiv_id']}")
    return result


# ======================================================
# STEP 7 — SECTION DETECTION (internal)
# ======================================================

def _norm_section(name: str) -> str:
    low = name.lower().strip().rstrip(":")
    return _SECTION_ALIASES.get(low, low)


def _detect_sections(full_text: str) -> List[SectionDict]:
    t0      = time.time()
    current = "unknown"
    out     = []
    for line in full_text.split("\n"):
        stripped = line.strip()
        is_h     = False
        if stripped and len(stripped) < 80:
            m = _SECTION_RE.match(stripped)
            if m:
                current = _norm_section(m.group(1))
                is_h    = True
        out.append({"section": current, "text": stripped, "is_heading": is_h})

    detected = sorted(set(s["section"] for s in out if s["section"] != "unknown"))
    print(f"[STEP 7] Sections: {len(out)} lines  "
          f"| {len(detected)} labels: {detected}  "
          f"in {time.time()-t0:.2f}s")
    return out


# ======================================================
# PUBLIC API
# ======================================================

def load_scientific_pdf(
    pdf_path: str | Path,
    verbose: bool = True
) -> RagPdfResult:
    """
    Load và phân tích một file PDF khoa học.

    Parameters
    ----------
    pdf_path : str | Path
        Đường dẫn đến file PDF.
    verbose  : bool
        Nếu True, in progress log ra stdout (mặc định True).

    Returns
    -------
    RagPdfResult
        Object chứa đầy đủ kết quả phân tích.
        Gọi .to_langchain_documents() để lấy List[Document] cho chunking.
        Gọi .summary() để in tóm tắt.

    Raises
    ------
    FileNotFoundError
        Nếu pdf_path không tồn tại.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if verbose:
        print(f"\n{'='*56}")
        print(f"  RAG PDF Loader v3.0  —  {path.name}")
        print(f"  Size: {path.stat().st_size / 1024:.1f} KB")
        print(f"{'='*56}")

    t_total = time.time()

    pages         = _parse_pages(path)
    layout        = _detect_layout(path)
    column_layout = _analyze_columns(layout)
    candidates    = _find_table_candidates(path, column_layout)
    tables        = _extract_tables(path, candidates)
    metadata      = _extract_metadata(path, layout)

    full_text = "\n".join(
        p["text"] for p in pages if not p.get("corrupted")
    )
    sections = _detect_sections(full_text)

    result = RagPdfResult(
        pages=pages,
        layout=layout,
        column_layout=column_layout,
        tables=tables,
        metadata=metadata,
        sections=sections,
    )

    if verbose:
        print(f"\n  Total pipeline time: {time.time()-t_total:.2f}s")
        print(result.summary())

    return result
