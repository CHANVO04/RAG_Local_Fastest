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
```test
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
```
---

## Bắt đầu nhanh (Quick Start) trên Window
Setup:
Bạn hãy cài đặt Python từ 8 trở lên (https://www.python.org/downloads/windows/)
Tải git
Sau đó mở Window PowerShell và
Làm theo các bước dưới đây để thiết lập môi trường và khởi chạy dự án trên máy tính của bạn:
```bash
Bước 1: Lấy dự án về từ github
git clone https://github.com/CHANVO04/RAG_Local_Fastest.git
cd RAG_Local_Fastest
Bước 2: Cài môi trường ảo:
python -m venv .venv
.venv\Scripts\activate.bat
Bước 3: Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
Bước 4: Cấu hình biến môi trường (API Key)
4.1) Tìm file .env.example trong thư mục dự án.
4.2) Đổi tên file này thành .env (hoặc copy/paste ra một file mới tên .env).
4.3)Mở file .env bằng trình chỉnh sửa văn bản (vd NotePad) và thêm API Key của OpenAI vào:
Đoạn mã
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
Bước 5: Khởi chạy ứng dụng
streamlit run app.py
