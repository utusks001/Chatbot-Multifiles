import os
from io import BytesIO
import requests
import streamlit as st
from dotenv import load_dotenv

# File parsing
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation as PptxPresentation
from PIL import Image
import pandas as pd  # ⬅️ NEW

# LangChain / VectorStore / Embeddings / LLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -------------------------
# Config / env
# -------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY")

st.set_page_config(
    page_title="Gemini + Groq Multi-file Chatbot (FAISS + OCR.Space + CSV/Excel)",
    page_icon="🤖",
    layout="wide"
)

if not (GOOGLE_API_KEY or GROQ_API_KEY):
    st.error("❌ GOOGLE_API_KEY atau GROQ_API_KEY tidak ditemukan. Tambahkan ke file .env sebelum menjalankan.")
    st.stop()

# Embeddings
EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

# -------------------------
# File extractors - teks dokumen
# -------------------------
def extract_text_from_pdf(file_bytes):
    text = ""
    try:
        reader = PdfReader(file_bytes)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        st.warning(f"⚠️ Gagal ekstrak PDF: {e}")
    return text

def extract_text_from_txt(file_bytes):
    try:
        return file_bytes.read().decode("utf-8", errors="ignore")
    except Exception as e:
        st.warning(f"⚠️ Gagal baca TXT: {e}")
        return ""

def extract_text_from_docx(file_bytes):
    text = ""
    try:
        file_bytes.seek(0)
        doc = DocxDocument(file_bytes)
        for p in doc.paragraphs:
            if p.text:
                text += p.text + "\n"
    except Exception as e:
        st.warning(f"⚠️ Gagal ekstrak DOCX: {e}")
    return text

def extract_text_from_pptx(file_bytes):
    text = ""
    try:
        file_bytes.seek(0)
        prs = PptxPresentation(file_bytes)
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    text += shape.text + "\n"
    except Exception as e:
        st.warning(f"⚠️ Gagal ekstrak PPTX: {e}")
    return text

# -------------------------
# OCR.Space Extractor (Image Files)
# -------------------------
def extract_text_from_image(file_bytes, filename="upload.png"):
    if not OCR_SPACE_API_KEY:
        st.warning("⚠️ OCR_SPACE_API_KEY tidak ditemukan di .env")
        return ""
    try:
        file_bytes.seek(0)
        response = requests.post(
            "https://api.ocr.space/parse/image",
            files={"file": (filename, file_bytes, "image/png")},
            data={"apikey": OCR_SPACE_API_KEY, "language": "eng"},
        )
        result = response.json()
        if result.get("IsErroredOnProcessing"):
            st.warning("⚠️ OCR.Space gagal: " + str(result.get("ErrorMessage", ["Unknown error"])))
            return ""
        text = "\n".join([p["ParsedText"] for p in result.get("ParsedResults", []) if "ParsedText" in p])
        return text.strip()
    except Exception as e:
        st.warning(f"⚠️ OCR.Space error: {e}")
        return ""

# -------------------------
# Helpers untuk DataFrame ➜ teks (untuk indexing + RAG)
# -------------------------
def df_profile_text(df, name="", sheet_name=None, max_rows_for_sample=200):
    try:
        # Profil ringkas
        rows, cols = df.shape
        dtypes = df.dtypes.astype(str).to_dict()
        missing = df.isna().sum().to_dict()
        # Statistik numerik (ringkas)
        stats = df.describe(include="all", datetime_is_numeric=True).transpose().reset_index().to_string(index=False)
        # Sampel baris (biar tidak meledak)
        sample = df.head(max_rows_for_sample)
        sample_csv = sample.to_csv(index=False)

        header = f"DATAFRAME SUMMARY — file={name}" + (f", sheet={sheet_name}" if sheet_name else "")
        block = [
            header,
            f"shape: {rows} rows x {cols} cols",
            f"dtypes: {dtypes}",
            f"missing_counts: {missing}",
            "describe():",
            stats,
            "sample(head):",
            sample_csv
        ]
        return "\n".join([str(x) for x in block if x is not None])
    except Exception as e:
        return f"[profiling error] {e}"

# -------------------------
# Ekstraktor CSV / Excel
# -------------------------
def extract_text_from_csv(file_bytes, filename):
    try:
        file_bytes.seek(0)
        # Upayakan encoding umum; fallback ke 'latin-1'
        try:
            df = pd.read_csv(file_bytes)
        except Exception:
            file_bytes.seek(0)
            df = pd.read_csv(file_bytes, encoding="latin-1")
        # simpan df ke session untuk preview
        st.session_state.dataframes[filename] = {"type": "csv", "sheets": {"CSV": df}}
        # return teks untuk index
        return df_profile_text(df, name=filename, sheet_name="CSV")
    except Exception as e:
        st.warning(f"⚠️ Gagal baca CSV: {e}")
        return ""

def extract_text_from_excel(file_bytes, filename):
    text_parts = []
    try:
        file_bytes.seek(0)
        xls = pd.ExcelFile(file_bytes)  # butuh openpyxl (xlsx) / xlrd (xls)
        sheet_map = {}
        for s in xls.sheet_names:
            try:
                df = xls.parse(s)
                sheet_map[s] = df
                text_parts.append(df_profile_text(df, name=filename, sheet_name=s))
            except Exception as se:
                st.warning(f"⚠️ Gagal parse sheet '{s}' di {filename}: {se}")
        if sheet_map:
            st.session_state.dataframes[filename] = {"type": "excel", "sheets": sheet_map}
        return "\n\n---\n\n".join(text_parts)
    except Exception as e:
        st.warning(f"⚠️ Gagal baca Excel: {e}")
        return ""

# -------------------------
# Generic extractor
# -------------------------
def extract_text_from_file(uploaded_file):
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    bio = BytesIO(raw)

    if name.endswith(".pdf"):
        return extract_text_from_pdf(bio)
    elif name.endswith(".txt"):
        return extract_text_from_txt(BytesIO(raw))
    elif name.endswith(".docx"):
        return extract_text_from_docx(BytesIO(raw))
    elif name.endswith(".pptx"):
        return extract_text_from_pptx(BytesIO(raw))
    elif name.endswith(".csv"):
        return extract_text_from_csv(BytesIO(raw), uploaded_file.name)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return extract_text_from_excel(BytesIO(raw), uploaded_file.name)
    elif name.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".jfif")):
        return extract_text_from_image(BytesIO(raw), filename=uploaded_file.name)
    elif name.endswith(".doc") or name.endswith(".ppt"):
        st.warning(f"⚠️ File `{uploaded_file.name}` berformat lama (.doc/.ppt). Silakan konversi ke .docx/.pptx.")
        return ""
    else:
        st.warning(f"⚠️ Tipe file `{uploaded_file.name}` tidak didukung.")
        return ""

# -------------------------
# Build documents & FAISS
# -------------------------
def build_documents_from_uploads(uploaded_files):
    docs = []
    for f in uploaded_files:
        text = extract_text_from_file(f)
        if not text or not text.strip():
            continue
        chunks = SPLITTER.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(Document(page_content=chunk, metadata={"source_file": f.name, "chunk_id": i}))
    return docs

def build_faiss_from_documents(docs):
    if not docs:
        return None
    vs = FAISS.from_documents(docs, embedding=EMBEDDINGS)
    return vs

# -------------------------
# Prompt formatting helpers
# -------------------------
def format_context(snippets):
    parts = []
    for idx, d in enumerate(snippets, start=1):
        src = d.metadata.get("source_file", "unknown")
        cid = d.metadata.get("chunk_id", "-")
        parts.append(f"[{idx}] ({src}#chunk-{cid})\n{d.page_content}")
    return "\n\n---\n\n".join(parts)

def render_sources(snippets):
    with st.expander("🔎 Sumber konteks yang dipakai"):
        for i, d in enumerate(snippets, start=1):
            src = d.metadata.get("source_file", "unknown")
            cid = d.metadata.get("chunk_id", "-")
            preview = d.page_content[:300].replace("\n", " ")
            st.markdown(f"**[{i}]** **{src}** (chunk {cid})")
            st.caption(preview + ("..." if len(d.page_content) > 300 else ""))

# -------------------------
# Streamlit UI
# -------------------------
st.title("🤖 Gemini 2.5 Flash + Groq — Multi-files + OCR.Space + CSV/XLS/XLSX")
st.write("Upload banyak file (PDF, TXT, DOCX, PPTX, Images, **CSV/XLS/XLSX**). Tabel akan dipreview & diprofiling, lalu ikut di-index untuk Q&A.")

# Sidebar
st.sidebar.header("📂 Upload & Build")
uploaded_files = st.sidebar.file_uploader(
    "Upload files (pdf, txt, docx, pptx, images, csv, xls, xlsx) — boleh banyak",
    type=["pdf", "txt", "docx", "pptx", "jpg", "jpeg", "png", "gif", "bmp", "jfif", "csv", "xls", "xlsx"],
    accept_multiple_files=True
)
build_btn = st.sidebar.button("🚀 Build Vector Store")
clear_btn = st.sidebar.button("🧹 Reset vector store")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "dataframes" not in st.session_state:
    st.session_state.dataframes = {}

if clear_btn:
    st.session_state.vector_store = None
    st.session_state.indexed_files = []
    st.session_state.dataframes = {}
    st.success("Vector store di-reset.")

if build_btn:
    if not uploaded_files:
        st.sidebar.warning("Silakan upload minimal 1 file terlebih dahulu.")
    else:
        with st.spinner("📦 Memproses file dan membuat vector store..."):
            docs = build_documents_from_uploads(uploaded_files)
            if not docs:
                st.sidebar.error("Tidak ada teks valid berhasil diekstrak. Periksa file.")
            else:
                vs = build_faiss_from_documents(docs)
                st.session_state.vector_store = vs
                st.session_state.indexed_files = [f.name for f in uploaded_files]
                st.sidebar.success(f"Vector store terbangun. Dokumen: {len(st.session_state.indexed_files)} | Chunk total: {len(docs)}")

# Show indexed files
if st.session_state.indexed_files:
    st.markdown("**Dokumen terindeks:**")
    st.write(" • " + "\n • ".join(st.session_state.indexed_files))

# -------------------------
# 📊 Data Preview & Profiling
# -------------------------
if st.session_state.dataframes:
    st.subheader("📊 Data Preview & Profiling")
    for fname, payload in st.session_state.dataframes.items():
        with st.expander(f"🔎 {fname}"):
            for sheet, df in payload["sheets"].items():
                st.markdown(f"**Sheet:** {sheet}")
                r, c = df.shape
                st.caption(f"shape: {r} rows × {c} cols")
                st.dataframe(df.head(50))
                # Ringkas tipe & missing
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**dtypes**")
                    st.write(pd.DataFrame({"dtype": df.dtypes.astype(str)}))
                with col2:
                    st.write("**missing (count)**")
                    st.write(df.isna().sum().to_frame("missing"))
                # Statistik numerik ringkas
                with st.expander("📈 describe()"):
                    try:
                        st.write(df.describe(include="all", datetime_is_numeric=True))
                    except Exception as e:
                        st.warning(f"describe() error: {e}")

# -------------------------
# Pilih LLM Provider
# -------------------------
model_choice = st.sidebar.radio(
    "Pilih LLM Provider:",
    ["Gemini 2.5 Flash (Google)", "Groq (llama-3.3-70b-versatile)"]
)

# -------------------------
# Query area
# -------------------------
prompt = st.text_input(
    "Tanyakan sesuatu berdasarkan dokumen/tabel yang diupload:",
    placeholder="Misal: Ringkas tren penjualan per wilayah 2024; atau 'kolom mana yang paling banyak missing?'"
)
ask_btn = st.button("Tanyakan")

if ask_btn:
    if not prompt.strip():
        st.warning("Masukkan pertanyaan terlebih dahulu.")
    elif st.session_state.vector_store is None:
        st.info("Belum ada vector store. Upload file dan klik 'Build Vector Store'.")
    else:
        with st.spinner("🔎 Mengambil konteks dari vector store..."):
            results = st.session_state.vector_store.similarity_search(prompt, k=5)

        context_text = format_context(results)
        system_instructions = (
            "Jawablah seakurat dan sedetil mungkin berdasarkan konteks berikut. "
            "Jika konteks berupa ringkasan tabel (CSV/Excel), gunakan metrik yang tersedia (shape, dtypes, missing, describe, sample). "
            "Jika jawaban tidak ada, katakan: "
            "\"Jawaban tidak tersedia dalam konteks yang diberikan, "
            "pelajari dan berikan tambahan informasi yang diperlukan sesuai konteks atau topik "
            "yang ditanyakan dari berbagai sumber atau referensi di internet dengan menyertakan link website.\" "
            "Sertakan referensi [angka] ke potongan konteks bila relevan."
        )

        composed_prompt = (
            f"{system_instructions}\n\n"
            f"=== KONTEX ===\n{context_text}\n\n"
            f"=== PERTANYAAN ===\n{prompt}\n\n"
            f"=== JAWABAN ==="
        )

        try:
            if model_choice.startswith("Gemini"):
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
                with st.spinner("🤖 Gemini sedang menjawab..."):
                    response = llm.invoke(composed_prompt)
            else:
                from langchain_groq import ChatGroq
                llm = ChatGroq(
                    temperature=0.2,
                    groq_api_key=GROQ_API_KEY,
                    model_name="llama-3.3-70b-versatile"
                )
                with st.spinner("⚡ Groq sedang menjawab..."):
                    response = llm.invoke(composed_prompt)

            st.subheader("💬 Jawaban")
            out_text = getattr(response, "content", None) or str(response)
            st.write(out_text)
            render_sources(results)

        except Exception as e:
            st.error(f"❌ Error saat memanggil LLM: {e}")
