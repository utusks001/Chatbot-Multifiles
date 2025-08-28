# 🤖 Gemini 2.5 Flash + Groq Chatbot — Multi-files + OCR.Space
(https://chatbot-multifiles-images-ocr.streamlit.app/)

Aplikasi **Streamlit Chatbot** yang mendukung:
- 📄 Upload multi-file: **PDF, TXT, DOCX, PPTX**
- 🖼️ Upload gambar: **JPG, PNG, GIF, BMP, JFIF** → parsing teks dengan **OCR.Space API**
- 🔍 **Vector Store**: FAISS + HuggingFace embeddings
- 🤖 Pilihan **LLM**:
  - Google **Gemini 2.5 Flash**
  - **Groq** (llama-3.3-70b-versatile atau Mixtral)

---

## 🚀 Fitur Utama
- Upload banyak file sekaligus
- Otomatis ekstrak teks dari PDF, Word, PPT, TXT, dan Gambar (OCR)
- Simpan teks ke dalam **FAISS Vector Store**
- Tanya jawab dengan dokumen menggunakan **Gemini** atau **Groq**


# Setup & Installation

## 1. Clone atau download repostiroy ini
git clone https://github.com/utusks001/Chatbot-Multifiles.git

## 2. Install Dependencies:
pip install -r requirements.txt
//Make sure you have Python 3.7+ installed.

## 3. Set Environment Variables: buat file dengan nama .env kemudian isi file tersebut dengan API Keys sebagai berikut Example:
- GROQ_API_KEY = "GROQQ API KEY"
- GOOGLE_API_KEY = "GOOGLE API KEY"

## 4. Run Streamlit di local
streamlit run main.py

## 5. Lalu deploy ke 
https://share.streamlit.io/
