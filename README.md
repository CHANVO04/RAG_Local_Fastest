# RAG Local Fastest

Hệ thống RAG local cho tài liệu PDF sử dụng **FAISS**, **FastEmbed**, **FlashRank**, **LangChain**, **OpenAI API** và **Streamlit**.  
Project hỗ trợ ingest tài liệu PDF, semantic chunking, semantic cache, reranking và hỏi đáp có trích nguồn ngay trên giao diện web.

## Tính năng
- Upload PDF trên giao diện Streamlit
- Ingest tài liệu vào FAISS
- Semantic chunking
- Rerank với FlashRank
- Semantic cache
- Hỏi đáp có trích nguồn
- Xem lịch sử truy vấn và logs
- Preview tài liệu đã ingest
- Xóa tài liệu khỏi registry và vector store

## Công nghệ sử dụng
- Python
- Streamlit
- FAISS
- FastEmbed
- FlashRank
- LangChain
- OpenAI API
- PyMuPDF
- Camelot

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt