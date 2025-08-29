import os
from io import BytesIO
import requests
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Document parsing
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation as PptxPresentation

# LangChain / VectorStore / Embeddings
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

st.set_page_config(page_title="Chatbot + Auto Analysis", page_icon="🤖", layout="wide")

# -------------------------
# Session state init
# -------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "dataframes" not in st.session_state:
    st.session_state.dataframes = {}

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("📂 Upload & Build")
uploaded_files = st.sidebar.file_uploader(
    "Upload files (pdf, txt, docx, pptx, images, csv, xls, xlsx) — boleh banyak",
    type=["pdf", "txt", "docx", "pptx", "jpg", "jpeg", "png", "gif", "bmp", "jfif", "csv", "xls", "xlsx"],
    accept_multiple_files=True
)

llm_choice = st.sidebar.radio(
    "Pilih LLM Provider:",
    ["Gemini 2.5 Flash (Google)", "Groq (llama-3.3-70b-versatile)"],
    index=0
)

build_btn = st.sidebar.button("🚀 Build Vector Store")
clear_btn = st.sidebar.button("🧹 Reset All")

# -------------------------
# Embeddings & splitter
# -------------------------
EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

# -------------------------
# Helpers
# -------------------------
def safe_describe(df):
    try:
        return df.describe(include="all", datetime_is_numeric=True)
    except TypeError:
        return df.describe(include="all")

def df_info_text(df):
    import io
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()

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
    except Exception:
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
        st.warning("⚠️ OCR_SPACE_API_KEY tidak ditemukan di .env — image OCR dinonaktifkan.")
        return ""
    try:
        file_bytes.seek(0)
        data = file_bytes.read()
        if len(data) == 0:
            st.warning(f"⚠️ Gambar {filename} kosong atau gagal terbaca.")
            return ""
        resp = requests.post(
            "https://api.ocr.space/parse/image",
            files={"file": (filename, BytesIO(data), "image/png")},
            data={"apikey": OCR_SPACE_API_KEY, "language": "eng"},
            timeout=60
        )
        result = resp.json()
        if result.get("IsErroredOnProcessing"):
            st.warning("⚠️ OCR.Space error: " + str(result.get("ErrorMessage", ["Unknown error"])))
            return ""
        parsed = [p.get("ParsedText", "") for p in result.get("ParsedResults", [])]
        return "\n".join(parsed).strip()
    except Exception as e:
        st.warning(f"⚠️ OCR error: {e}")
        return ""

# CSV / Excel
def extract_text_from_csv(file_bytes, filename):
    try:
        file_bytes.seek(0)
        df = pd.read_csv(file_bytes)
        st.session_state.dataframes[filename] = {"sheets": {"CSV": df}}
        return df_to_index_text(df, filename, "CSV")
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
                text_parts.append(df_to_index_text(df, filename, s))
            except Exception as se:
                st.warning(f"⚠️ Gagal parse sheet '{s}' di {filename}: {se}")
        if sheet_map:
            st.session_state.dataframes[filename] = {"sheets": sheet_map}
        return "\n\n---\n\n".join(text_parts)
    except Exception as e:
        st.warning(f"⚠️ Gagal baca Excel: {e}")
        return ""

def df_to_index_text(df, filename, sheet_name):
    rows, cols = df.shape
    stats = safe_describe(df).transpose().reset_index().to_string(index=False)
    sample = df.head(20).to_csv(index=False)
    return f"DATAFRAME — file={filename}, sheet={sheet_name}\nshape: {rows}x{cols}\n{stats}\nSAMPLE:\n{sample}"

def extract_text_from_file(uploaded_file):
    name = uploaded_file.name
    lname = name.lower()
    raw = uploaded_file.read()
    bio = BytesIO(raw)

    if lname.endswith(".pdf"):
        return extract_text_from_pdf(bio)
    if lname.endswith(".txt"):
        return extract_text_from_txt(BytesIO(raw))
    if lname.endswith(".docx"):
        return extract_text_from_docx(BytesIO(raw))
    if lname.endswith(".pptx"):
        return extract_text_from_pptx(BytesIO(raw))
    if lname.endswith(".csv"):
        return extract_text_from_csv(BytesIO(raw), name)
    if lname.endswith((".xlsx", ".xls")):
        return extract_text_from_excel(BytesIO(raw), name)
    if lname.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".jfif")):
        return extract_text_from_image(BytesIO(raw), filename=name)
    st.warning(f"⚠️ Format `{name}` tidak didukung.")
    return ""

# -------------------------
# Build documents & FAISS
# -------------------------
def build_documents_from_uploads(files):
    docs = []
    for f in files:
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
# Auto-analysis
# -------------------------
def auto_analyze_dataframe(df, filename, sheet_name, show_in_app=True):
    num_df = df.select_dtypes(include="number")

    # Export Excel
    out_excel = BytesIO()
    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Data", index=False)
        try:
            safe_describe(df).to_excel(writer, sheet_name="Describe")
        except:
            pass
        if not num_df.empty:
            num_df.corr().to_excel(writer, sheet_name="Correlation")
    out_excel.seek(0)

    # Export HTML
    html_report = f"<h2>Analysis — {filename}/{sheet_name}</h2><pre>{df.head(20).to_string()}</pre>"
    html_bytes = html_report.encode("utf-8")

    if show_in_app:
        st.markdown(f"### 📄 Analisa: {filename} — {sheet_name}")
        st.write("**Head (10):**")
        st.dataframe(df.head(10))
        st.write("**Tail (10):**")
        st.dataframe(df.tail(10))

        st.write("**describe():**")
        st.dataframe(safe_describe(df))

        st.write("**info():**")
        st.text(df_info_text(df))

        # Outlier Detection (Boxplot) untuk Sales, Quantity, Profit
        target_cols = [c for c in ["Sales", "Quantity", "Profit"] if c in df.columns]
        if target_cols:
            st.write("**Outlier Detection (Boxplot):**")
            for col in target_cols:
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.boxplot(x=df[col], ax=ax)
                ax.set_title(f"Outliers — {col}")
                st.pyplot(fig)

        # Scatter plot Sales vs Profit
        if "Sales" in df.columns and "Profit" in df.columns:
            st.write("**Scatter Plot: Sales vs Profit**")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.scatterplot(x=df["Sales"], y=df["Profit"], ax=ax)
            ax.set_xlabel("Sales")
            ax.set_ylabel("Profit")
            st.pyplot(fig)

        # Correlation heatmap
        if not num_df.empty:
            corr = num_df.corr()
            st.write("**Correlation matrix:**")
            st.dataframe(corr)
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    # Download buttons
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Download Excel", out_excel, f"analysis_{filename}_{sheet_name}.xlsx")
    with c2:
        st.download_button("⬇️ Download HTML", html_bytes, f"analysis_{filename}_{sheet_name}.html", mime="text/html")

# -------------------------
# Main
# -------------------------
st.title("🤖 Multi-file Chatbot + Auto Analysis (final)")

if uploaded_files:
    for f in uploaded_files:
        if f.name not in st.session_state.dataframes:
            extract_text_from_file(f)

if build_btn and uploaded_files:
    with st.spinner("Membangun vector store..."):
        docs = build_documents_from_uploads(uploaded_files)
        st.session_state.vector_store = build_faiss_from_documents(docs)
        st.session_state.indexed_files = [f.name for f in uploaded_files]
        st.success(f"Vector store terbangun ({len(docs)} chunks).")

if clear_btn:
    st.session_state.vector_store = None
    st.session_state.indexed_files = []
    st.session_state.dataframes = {}
    st.sidebar.success("Reset selesai.")

# Analysis
if st.session_state.dataframes:
    st.subheader("📊 Data Preview & Analisa")
    show_in_app = st.checkbox("Tampilkan analisa di Streamlit", True)
    for fname, payload in st.session_state.dataframes.items():
        with st.expander(f"File: {fname}", expanded=False):
            for sheet, df in payload["sheets"].items():
                auto_analyze_dataframe(df, fname, sheet, show_in_app)

# Query
st.subheader("💬 Ajukan Pertanyaan")
prompt = st.text_input("Pertanyaan berdasarkan dokumen:")
if st.button("Tanyakan"):
    if not prompt.strip():
        st.warning("Masukkan pertanyaan dulu")
    elif st.session_state.vector_store is None:
        st.info("Belum ada vector store. Upload file & klik Build Vector Store.")
    else:
        results = st.session_state.vector_store.similarity_search(prompt, k=5)
        ctx = "\n\n".join([d.page_content for d in results])
        q_prompt = f"Jawablah berdasarkan konteks berikut:\n{ctx}\n\nPertanyaan: {prompt}\nJawaban:"
        try:
            if llm_choice.startswith("Gemini"):
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
                response = llm.invoke(q_prompt)
            else:
                from langchain_groq import ChatGroq
                llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY, temperature=0.2)
                response = llm.invoke(q_prompt)
            st.write(getattr(response, "content", str(response)))
        except Exception as e:
            st.error(f"LLM error: {e}")
