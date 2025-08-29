# app.py
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

# Data analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# LangChain / VectorStore / Embeddings / LLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -------------------------
# Config
# -------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY")

st.set_page_config(
    page_title="Chatbot + Excel Analysis",
    page_icon="🤖",
    layout="wide"
)

if not (GOOGLE_API_KEY or GROQ_API_KEY):
    st.error("❌ GOOGLE_API_KEY atau GROQ_API_KEY tidak ditemukan. Tambahkan ke file .env sebelum menjalankan.")
    st.stop()

# -------------------------
# Embeddings & splitter
# -------------------------
EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

# -------------------------
# Session state
# -------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "dataframes" not in st.session_state:
    st.session_state.dataframes = {}

# -------------------------
# DataFrame helpers
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

def df_to_index_text(df, filename, sheet_name):
    rows, cols = df.shape
    stats = safe_describe(df).transpose().reset_index().to_string(index=False)
    sample = df.head(20).to_csv(index=False)
    return f"DATAFRAME — file={filename}, sheet={sheet_name}\nshape: {rows}x{cols}\n{stats}\nSAMPLE:\n{sample}"

# -------------------------
# Extractors
# -------------------------
def extract_text_from_pdf(file_bytes: BytesIO):
    text = ""
    try:
        reader = PdfReader(file_bytes)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    except Exception as e:
        st.warning(f"⚠️ Gagal ekstrak PDF: {e}")
    return text

def extract_text_from_txt(file_bytes: BytesIO):
    try:
        return file_bytes.read().decode("utf-8", errors="ignore")
    except Exception as e:
        st.warning(f"⚠️ Gagal baca TXT: {e}")
        return ""

def extract_text_from_docx(file_bytes: BytesIO):
    text = ""
    try:
        doc = DocxDocument(file_bytes)
        for p in doc.paragraphs:
            if p.text:
                text += p.text + "\n"
    except Exception as e:
        st.warning(f"⚠️ Gagal ekstrak DOCX: {e}")
    return text

def extract_text_from_pptx(file_bytes: BytesIO):
    text = ""
    try:
        prs = PptxPresentation(file_bytes)
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    text += shape.text + "\n"
    except Exception as e:
        st.warning(f"⚠️ Gagal ekstrak PPTX: {e}")
    return text

def extract_text_from_image(file_bytes: BytesIO, filename="upload.png"):
    if not OCR_SPACE_API_KEY:
        st.warning("⚠️ OCR_SPACE_API_KEY tidak ditemukan di .env — image OCR dinonaktifkan.")
        return ""
    try:
        file_bytes.seek(0)
        data = file_bytes.read()
        if not data:
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
            st.warning("⚠️ OCR.Space error: " + str(result.get("ErrorMessage", ['Unknown error'])))
            return ""
        parsed = [p.get("ParsedText", "") for p in result.get("ParsedResults", [])]
        return "\n".join(parsed).strip()
    except Exception as e:
        st.warning(f"⚠️ OCR error: {e}")
        return ""

def extract_text_from_csv(file_bytes: BytesIO, filename: str):
    try:
        df = pd.read_csv(file_bytes)
        st.session_state.dataframes[filename] = {"sheets": {"CSV": df}}
        return df_to_index_text(df, filename, "CSV")
    except Exception as e:
        st.warning(f"⚠️ Gagal baca CSV {filename}: {e}")
        return ""

def extract_text_from_excel(file_bytes: BytesIO, filename: str):
    text_parts = []
    try:
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
        st.warning(f"⚠️ Gagal baca Excel {filename}: {e}")
        return ""

# -------------------------
# Universal file dispatcher (FIXED)
# -------------------------
def extract_text_from_file(uploaded_file):
    name = uploaded_file.name
    lname = name.lower()
    raw = uploaded_file.getvalue()  # ✅ fix: always fresh bytes

    if lname.endswith(".pdf"):
        return extract_text_from_pdf(BytesIO(raw))
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

    st.warning(f"⚠️ Tipe file `{uploaded_file.name}` tidak didukung.")
    return ""

# -------------------------
# Build docs & FAISS
# -------------------------
def build_documents_from_uploads(uploaded_files):
    docs = []
    for f in uploaded_files:
        text = extract_text_from_file(f)
        if text.strip():
            chunks = SPLITTER.split_text(text)
            for i, chunk in enumerate(chunks):
                docs.append(Document(page_content=chunk, metadata={"source": f.name, "chunk_id": i}))
    return docs

def build_faiss_from_documents(docs):
    if not docs:
        return None
    return FAISS.from_documents(docs, embedding=EMBEDDINGS)

# -------------------------
# Data analysis (Excel/CSV)
# -------------------------
def auto_analyze_dataframe(df: pd.DataFrame, filename: str, sheet_name: str):
    num_df = df.select_dtypes(include="number")

    st.markdown(f"### 📄 Analisa: {filename} — {sheet_name}")
    st.dataframe(df.head(10))
    st.dataframe(df.tail(10))
    st.dataframe(safe_describe(df))
    st.text(df_info_text(df))

    target_cols = [c for c in ["Sales", "Quantity", "Profit"] if c in df.columns]
    if target_cols:
        st.write("**Outlier (Boxplot)**")
        for col in target_cols:
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.boxplot(x=df[col].dropna(), ax=ax)
            st.pyplot(fig)

    if "Sales" in df.columns and "Profit" in df.columns:
        st.write("**Scatter Sales vs Profit**")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.scatterplot(x=df["Sales"], y=df["Profit"], ax=ax)
        st.pyplot(fig)

    if not num_df.empty:
        st.write("**Correlation Heatmap**")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# -------------------------
# UI
# -------------------------
st.title("🤖 Chatbot + Multi-files + Excel Analysis")

uploaded_files = st.sidebar.file_uploader(
    "Upload files", 
    type=["pdf","txt","docx","pptx","jpg","jpeg","png","gif","bmp","jfif","csv","xls","xlsx"],
    accept_multiple_files=True
)

if st.sidebar.button("🚀 Build Vector Store"):
    if uploaded_files:
        with st.spinner("Membangun vector store..."):
            docs = build_documents_from_uploads(uploaded_files)
            st.session_state.vector_store = build_faiss_from_documents(docs)
            st.session_state.indexed_files = [f.name for f in uploaded_files]
        st.sidebar.success("✅ Vector store terbangun")

if st.session_state.indexed_files:
    st.write("**Dokumen terindeks:**")
    st.write(st.session_state.indexed_files)

if st.session_state.dataframes:
    st.subheader("📊 Analisa Excel/CSV")
    for fname, payload in st.session_state.dataframes.items():
        for sheet, df in payload["sheets"].items():
            auto_analyze_dataframe(df, fname, sheet)

# Q&A
model_choice = st.sidebar.radio("LLM Provider:", ["Gemini 2.5 Flash", "Groq Llama"])
query = st.text_input("Tanyakan sesuatu:")
if st.button("Tanyakan") and query:
    if st.session_state.vector_store:
        results = st.session_state.vector_store.similarity_search(query, k=5)
        context = "\n\n".join([d.page_content for d in results])
        prompt = f"Konteks:\n{context}\n\nPertanyaan:\n{query}"
        try:
            if model_choice.startswith("Gemini"):
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
                resp = llm.invoke(prompt)
            else:
                from langchain_groq import ChatGroq
                llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY, temperature=0.2)
                resp = llm.invoke(prompt)
            st.subheader("💬 Jawaban")
            st.write(getattr(resp, "content", str(resp)))
        except Exception as e:
            st.error(f"❌ LLM error: {e}")
