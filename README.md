# 🤖 Gemini + Groq Multi-file Chatbot (FAISS + OCR.Space)

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

---

## 📦 Instalasi Lokal

1. **Clone repo:**
   ```bash
   git clone https://github.com/username/gemini-groq-chatbot.git
   cd gemini-groq-chatbot
