# RAG Local Fastest

Hệ thống RAG local cho tài liệu PDF sử dụng **FAISS**, **FastEmbed**, **FlashRank**, **LangChain**, **OpenAI API** và **Streamlit**.

Project hỗ trợ:
- ingest tài liệu PDF
- semantic chunking
- semantic retrieval
- reranking kết quả truy xuất
- semantic cache
- hỏi đáp có trích nguồn ngay trên giao diện web

---

## Mục tiêu project

Project này được xây dựng để tạo một hệ thống hỏi đáp trên tài liệu PDF theo hướng **RAG (Retrieval-Augmented Generation)**, trong đó:

- tài liệu PDF được tải lên và xử lý cục bộ
- nội dung được chia thành các chunk phù hợp cho truy xuất
- vector được lưu trong **FAISS**
- kết quả truy xuất được **rerank** bằng **FlashRank**
- mô hình ngôn ngữ sử dụng **OpenAI API** để sinh câu trả lời
- người dùng có thể thao tác trực tiếp qua giao diện **Streamlit**

Project phù hợp cho:
- học và nghiên cứu về RAG
- thử nghiệm pipeline ingest và retrieval trên PDF
- xây dựng demo hỏi đáp tài liệu khoa học hoặc tài liệu nội bộ

---

## Tính năng

- Upload PDF trên giao diện Streamlit
- Ingest tài liệu vào FAISS
- Semantic chunking
- Semantic retrieval
- Rerank với FlashRank
- Semantic cache
- Hỏi đáp có trích nguồn
- Xem lịch sử truy vấn và logs
- Preview tài liệu đã ingest
- Xóa tài liệu khỏi registry và vector store

---

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
- NumPy
- Pandas
- python-dotenv

---

## Cấu trúc thư mục

```text
RAG_Local_Fastest/
├── app.py
├── Ingest_Local_Fastest.py
├── Query_Local_Fastest.py
├── rag_pdf_loader.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
│
├── data/
│   └── .gitkeep
│
├── vector_store/
│   └── .gitkeep
│
├── docs/
└── assets/
