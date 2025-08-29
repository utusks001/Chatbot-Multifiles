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
import pandas as pd
import matplotlib.pyplot as plt

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
    page_title="Gemini + Groq Multi-file Chatbot (Flexible Top-N)",
    page_icon="🤖",
    layout="wide"
)

if not (GOOGLE_API_KEY or GROQ_API_KEY):
    st.error("❌ GOOGLE_API_KEY atau GROQ_API_KEY tidak ditemukan di .env. Tambahkan setidaknya salah satu sebelum menjalankan.")
    st.stop()

# Embeddings & splitter
EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

# -------------------------
# Utility / compatibility
# -------------------------
def safe_describe(df):
    """Fallback describe untuk semua versi pandas"""
    try:
        return df.describe(include="all", datetime_is_numeric=True)
    except TypeError:
        return df.describe(include="all")

# -------------------------
# File extractors
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
            timeout=60
        )
        result = response.json()
        if result.get("IsErroredOnProcessing"):
            st.warning("⚠️ OCR.Space gagal: " + str(result.get("ErrorMessage", ["Unknown error"])))
            return ""
        text = "\n".join([p.get("ParsedText","") for p in result.get("ParsedResults", [])])
        return text.strip()
    except Exception as e:
        st.warning(f"⚠️ OCR.Space error: {e}")
        return ""

# -------------------------
# DataFrame helpers (profiling -> text for indexing)
# -------------------------
def df_profile_text(df, name="", sheet_name=None):
    rows, cols = df.shape
    dtypes = df.dtypes.astype(str).to_dict()
    missing = df.isna().sum().to_dict()
    stats = safe_describe(df).transpose().reset_index().to_string(index=False)
    sample_csv = df.head(20).to_csv(index=False)

    header = f"DATAFRAME SUMMARY — file={name}" + (f", sheet={sheet_name}" if sheet_name else "")
    block = [
        header,
        f"shape: {rows} rows x {cols} cols",
        f"dtypes: {dtypes}",
        f"missing_counts: {missing}",
        "describe():",
        stats,
        "sample(head 20):",
        sample_csv
    ]
    return "\n".join([str(x) for x in block if x is not None])

def extract_text_from_csv(file_bytes, filename):
    try:
        file_bytes.seek(0)
        try:
            df = pd.read_csv(file_bytes)
        except Exception:
            file_bytes.seek(0)
            df = pd.read_csv(file_bytes, encoding="latin-1")
        st.session_state.dataframes[filename] = {"type": "csv", "sheets": {"CSV": df}}
        return df_profile_text(df, name=filename, sheet_name="CSV")
    except Exception as e:
        st.warning(f"⚠️ Gagal baca CSV: {e}")
        return ""

def extract_text_from_excel(file_bytes, filename):
    text_parts = []
    try:
        file_bytes.seek(0)
        xls = pd.ExcelFile(file_bytes)
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
# Auto Analysis for DataFrame (dengan Top-N fleksibel)
# -------------------------
def auto_analyze_dataframe(df, name="", sheet_name=None, show_in_app=True):
    """
    Menampilkan ringkasan & menyediakan dropdown untuk memilih kolom apa yang
    ingin ditampilkan Top-N (descending). Juga membuat file export.
    """
    report_str = []
    rows, cols = df.shape

    # Basic summary text
    report_str.append(f"Dataset shape: {rows} rows × {cols} cols")
    report_str.append("Tipe data:\n" + str(df.dtypes.astype(str)))
    report_str.append("Missing values:\n" + str(df.isna().sum()))
    
    # Correlation (numerical)
    num_cols = df.select_dtypes(include="number")
    if not num_cols.empty:
        report_str.append("Correlation matrix:\n" + str(num_cols.corr()))
    
    # Trend waktu (cari column yang mengandung date/time/tanggal)
    for col in df.columns:
        if any(x in col.lower() for x in ["date", "time", "tanggal"]):
            try:
                df[col] = pd.to_datetime(df[col])
                trend = df.groupby(df[col].dt.to_period("M")).size()
                report_str.append(f"Trend waktu berdasarkan {col}:\n{trend}")
                break
            except Exception:
                pass

    # Outliers (IQR) for numeric cols
    if not num_cols.empty:
        for c in num_cols.columns:
            q1 = num_cols[c].quantile(0.25)
            q3 = num_cols[c].quantile(0.75)
            iqr = q3 - q1
            outliers = num_cols[(num_cols[c] < q1 - 1.5*iqr) | (num_cols[c] > q3 + 1.5*iqr)]
            if not outliers.empty:
                report_str.append(f"Outliers di kolom {c}:\n{outliers.head(20)}")

    report_text = "\n\n".join(report_str)

    # Prepare Excel export (Data + Summary + Correlation)
    output_excel = BytesIO()
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        try:
            df.to_excel(writer, index=False, sheet_name="Data")
        except Exception:
            # fallback jika nama sheet bermasalah
            df.reset_index().to_excel(writer, index=False, sheet_name="Data")
        safe_describe(df).to_excel(writer, sheet_name="Summary")
        if not num_cols.empty:
            num_cols.corr().to_excel(writer, sheet_name="Correlation")
    output_excel.seek(0)

    # Simple HTML export
    output_html = f"<html><body><pre>{report_text}</pre></body></html>".encode("utf-8")

    # UI display
    if show_in_app:
        st.write("**Preview data (head 10):**")
        st.dataframe(df.head(10))

        st.write("**Ringkasan analisa otomatis:**")
        st.text(report_text)

        # Automatic top-10 for common columns (Sales/Profit) for backward compatibility
        for candidate in ["sales", "profit"]:
            for col in df.columns:
                if col.lower() == candidate:
                    try:
                        top10 = df.sort_values(by=col, ascending=False).head(10)
                        st.markdown(f"**Top 10 berdasarkan {col} (descending):**")
                        st.dataframe(top10)
                    except Exception:
                        pass

        # ========== FLEXIBLE TOP-N UI ==========
        # Build widget keys unique per file+sheet
        widget_col_key = f"topcol_{name}_{sheet_name}"
        widget_n_key = f"topn_{name}_{sheet_name}"
        widget_btn_key = f"topbtn_{name}_{sheet_name}"

        cols_for_dropdown = list(df.columns)
        if cols_for_dropdown:
            st.markdown("----")
            st.markdown("### Pilih kolom untuk melihat Top-N (descending)")
            chosen_col = st.selectbox("Pilih kolom:", options=cols_for_dropdown, key=widget_col_key)
            max_n = min(100, max(1, len(df)))
            chosen_n = st.slider("Jumlah baris (N):", min_value=1, max_value=max_n, value=min(10, max_n), key=widget_n_key)
            if st.button("Tampilkan Top-N", key=widget_btn_key):
                try:
                    # Prioritaskan sorting numeric desc; if column non-numeric then show top value_counts
                    if pd.api.types.is_numeric_dtype(df[chosen_col]):
                        topn_df = df.sort_values(by=chosen_col, ascending=False).head(chosen_n)
                        st.markdown(f"**Top {chosen_n} berdasarkan {chosen_col} (numeric, descending):**")
                        st.dataframe(topn_df)
                    else:
                        # for non-numeric, show top frequent rows by value_counts on the column,
                        # then show those rows from original df (head per value)
                        vc = df[chosen_col].value_counts().head(chosen_n)
                        st.markdown(f"**Top {chosen_n} nilai terbanyak di kolom {chosen_col}:**")
                        st.dataframe(vc.to_frame(name="count"))
                        # optionally show sample rows for top values
                        top_values = vc.index.tolist()
                        sample_rows = df[df[chosen_col].isin(top_values)].head(50)
                        st.markdown(f"**Sample rows for top values of {chosen_col}:**")
                        st.dataframe(sample_rows)
                except Exception as e:
                    st.warning(f"⚠️ Gagal menampilkan Top-N untuk kolom {chosen_col}: {e}")

    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download laporan Excel",
            data=output_excel,
            file_name=f"analysis_{name}_{sheet_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    with col2:
        st.download_button(
            "⬇️ Download laporan HTML",
            data=output_html,
            file_name=f"analysis_{name}_{sheet_name}.html",
            mime="text/html"
        )

# -------------------------
# Streamlit UI main
# -------------------------
st.title("🤖 Gemini + Groq — Multi-file + Auto-Analysis + Flexible Top-N")
st.write("Upload file (PDF, TXT, DOCX, PPTX, Images, CSV, XLS, XLSX). Data tabular dianalisa otomatis. Pilih kolom mana yang ingin dilihat Top-N.")

# Sidebar: upload & build & LLM choice
st.sidebar.header("📂 Upload & Build")
uploaded_files = st.sidebar.file_uploader(
    "Upload files (boleh banyak)",
    type=["pdf", "txt", "docx", "pptx", "jpg", "jpeg", "png", "gif", "bmp", "jfif", "csv", "xls", "xlsx"],
    accept_multiple_files=True
)
build_btn = st.sidebar.button("🚀 Build Vector Store")
clear_btn = st.sidebar.button("🧹 Reset Semua")

llm_choice = st.sidebar.radio(
    "Pilih LLM Provider:",
    ["Gemini 2.5 Flash (Google)", "Groq (llama-3.3-70b-versatile)"],
    index=0
)

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
    st.success("✅ Semua state direset.")

# Build vector store action
if build_btn:
    if not uploaded_files:
        st.sidebar.warning("Silakan upload terlebih dahulu minimal 1 file.")
    else:
        with st.spinner("📦 Memproses file dan membangun vector store..."):
            docs = build_documents_from_uploads(uploaded_files)
            if not docs:
                st.sidebar.error("Tidak ada teks valid berhasil diekstrak — periksa file yang diupload.")
            else:
                vs = build_faiss_from_documents(docs)
                st.session_state.vector_store = vs
                st.session_state.indexed_files = [f.name for f in uploaded_files]
                st.sidebar.success(f"Vector store siap — {len(st.session_state.indexed_files)} file terindeks, {len(docs)} chunk.")

# Show indexed files
if st.session_state.indexed_files:
    st.markdown("**📚 Dokumen terindeks:**")
    st.write(" • " + "\n • ".join(st.session_state.indexed_files))
else:
    st.info("Belum ada dokumen terindeks. Upload file dan klik 'Build Vector Store' jika ingin pakai fitur tanya jawab.")

# Data preview & analysis
if st.session_state.dataframes:
    st.subheader("📊 Data Preview, Profiling & Analisa Otomatis")
    show_in_app = st.checkbox("Tampilkan analisa di Streamlit (per sheet)", value=True)
    for fname, payload in st.session_state.dataframes.items():
        with st.expander(f"🔎 File: {fname}"):
            for sheet_name, df in payload["sheets"].items():
                st.markdown(f"**Sheet:** {sheet_name}")
                auto_analyze_dataframe(df, name=fname, sheet_name=sheet_name, show_in_app=show_in_app)

# -------------------------
# Query area (selalu tampil)
# -------------------------
st.subheader("💬 Ajukan Pertanyaan")
prompt = st.text_input("Tanyakan sesuatu berdasarkan dokumen/tabel yang diupload:",
                       placeholder="Misal: Ringkas tren penjualan per wilayah 2024")
ask_btn = st.button("Tanyakan")

if ask_btn:
    if not prompt.strip():
        st.warning("Masukkan pertanyaan terlebih dahulu.")
    elif st.session_state.vector_store is None:
        st.info("Belum ada vector store. Upload file dan klik 'Build Vector Store' jika ingin jawaban berdasarkan dokumen.")
    else:
        with st.spinner("🔎 Mengambil konteks dari vector store..."):
            results = st.session_state.vector_store.similarity_search(prompt, k=5)

        context_text = "\n\n".join([d.page_content for d in results])
        composed_prompt = (
            "Jawablah seakurat mungkin berdasarkan konteks berikut.\n\n"
            f"=== KONTEX ===\n{context_text}\n\n"
            f"=== PERTANYAAN ===\n{prompt}\n\n"
            f"=== JAWABAN ==="
        )

        try:
            if llm_choice.startswith("Gemini"):
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

        except Exception as e:
            st.error(f"❌ Error saat memanggil LLM: {e}")
