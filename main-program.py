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
# Config
# -------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY")

st.set_page_config(page_title="Chatbot + Auto Analysis", page_icon="🤖", layout="wide")

if not (GOOGLE_API_KEY or GROQ_API_KEY):
    st.error("❌ GOOGLE_API_KEY atau GROQ_API_KEY tidak ditemukan di .env.")
    st.stop()

# Embeddings & splitter
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

# -------------------------
# File Extractors
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
    except:
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
        st.warning("⚠️ OCR_SPACE_API_KEY tidak ditemukan")
        return ""
    try:
        file_bytes.seek(0)
        resp = requests.post(
            "https://api.ocr.space/parse/image",
            files={"file": (filename, file_bytes, "image/png")},
            data={"apikey": OCR_SPACE_API_KEY, "language": "eng"},
            timeout=60
        )
        result = resp.json()
        if result.get("IsErroredOnProcessing"):
            return ""
        return "\n".join([p.get("ParsedText","") for p in result.get("ParsedResults", [])]).strip()
    except Exception as e:
        st.warning(f"⚠️ OCR error: {e}")
        return ""

# DataFrame summary text
def df_profile_text(df, name="", sheet_name=None):
    rows, cols = df.shape
    dtypes = df.dtypes.astype(str).to_dict()
    missing = df.isna().sum().to_dict()
    stats = safe_describe(df).transpose().reset_index().to_string(index=False)
    sample_csv = df.head(20).to_csv(index=False)
    return f"SUMMARY {name} {sheet_name}\nshape: {rows}x{cols}\n{dtypes}\nmissing:{missing}\n{stats}\n{sample_csv}"

def extract_text_from_csv(file_bytes, filename):
    try:
        file_bytes.seek(0)
        df = pd.read_csv(file_bytes)
        st.session_state.dataframes[filename] = {"sheets": {"CSV": df}}
        return df_profile_text(df, name=filename, sheet_name="CSV")
    except Exception as e:
        st.warning(f"⚠️ CSV error: {e}")
        return ""

def extract_text_from_excel(file_bytes, filename):
    parts = []
    try:
        file_bytes.seek(0)
        xls = pd.ExcelFile(file_bytes)
        sheets = {}
        for s in xls.sheet_names:
            try:
                df = xls.parse(s)
                sheets[s] = df
                parts.append(df_profile_text(df, name=filename, sheet_name=s))
            except Exception as e:
                st.warning(f"⚠️ Sheet {s} gagal: {e}")
        if sheets:
            st.session_state.dataframes[filename] = {"sheets": sheets}
        return "\n\n".join(parts)
    except Exception as e:
        st.warning(f"⚠️ Excel error: {e}")
        return ""

def extract_text_from_file(uploaded_file):
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    bio = BytesIO(raw)
    if name.endswith(".pdf"): return extract_text_from_pdf(bio)
    if name.endswith(".txt"): return extract_text_from_txt(BytesIO(raw))
    if name.endswith(".docx"): return extract_text_from_docx(BytesIO(raw))
    if name.endswith(".pptx"): return extract_text_from_pptx(BytesIO(raw))
    if name.endswith(".csv"): return extract_text_from_csv(BytesIO(raw), uploaded_file.name)
    if name.endswith((".xlsx",".xls")): return extract_text_from_excel(BytesIO(raw), uploaded_file.name)
    if name.endswith((".jpg",".jpeg",".png",".gif",".bmp",".jfif")): return extract_text_from_image(BytesIO(raw), uploaded_file.name)
    st.warning(f"⚠️ Format {name} tidak didukung")
    return ""

# -------------------------
# Build docs & vector store
# -------------------------
def build_documents_from_uploads(files):
    docs=[]
    for f in files:
        text=extract_text_from_file(f)
        if not text: continue
        for i,ch in enumerate(SPLITTER.split_text(text)):
            docs.append(Document(page_content=ch, metadata={"source":f.name,"chunk":i}))
    return docs

def build_faiss(docs):
    return FAISS.from_documents(docs, embedding=EMBEDDINGS) if docs else None

# -------------------------
# Auto analysis dataframe
# -------------------------
def auto_analyze_dataframe(df, name="", sheet_name=None):
    st.markdown("#### 🔍 Preview DataFrame")

    # Head / Tail
    st.write("**Head (10):**")
    st.dataframe(df.head(10))
    st.write("**Tail (10):**")
    st.dataframe(df.tail(10))

    # Describe
    st.write("**describe():**")
    st.dataframe(safe_describe(df))

    # Info
    import io
    buf=io.StringIO(); df.info(buf=buf)
    st.write("**info():**")
    st.text(buf.getvalue())

    # Flexible Top/Bottom N
    st.markdown("----")
    st.markdown("### Pilih kolom untuk Top/Bottom-N")
    col=st.selectbox("Kolom:", df.columns, key=f"col_{name}_{sheet_name}")
    n=st.slider("Jumlah baris:",1,min(100,len(df)),10,key=f"n_{name}_{sheet_name}")
    order=st.radio("Urutan:",["Top N (descending)","Bottom N (ascending)"],horizontal=True,key=f"ord_{name}_{sheet_name}")
    if st.button("Tampilkan",key=f"btn_{name}_{sheet_name}"):
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                asc=(order=="Bottom N (ascending)")
                st.dataframe(df.sort_values(by=col,ascending=asc).head(n))
            else:
                vc=df[col].value_counts(ascending=(order=="Bottom N (ascending)")).head(n)
                st.dataframe(vc.to_frame("count"))
        except Exception as e:
            st.warning(f"⚠️ Error: {e}")

    # Download
    out=BytesIO()
    with pd.ExcelWriter(out,engine="openpyxl") as w:
        df.to_excel(w,index=False,sheet_name="Data")
        safe_describe(df).to_excel(w,sheet_name="Summary")
    out.seek(0)
    st.download_button("⬇️ Download Excel",data=out,file_name=f"analysis_{name}_{sheet_name}.xlsx")

# -------------------------
# UI
# -------------------------
st.title("🤖 Multi-file Chatbot + Auto Analysis")

uploaded=st.sidebar.file_uploader("Upload files",type=["pdf","txt","docx","pptx","csv","xls","xlsx","jpg","jpeg","png","gif","bmp","jfif"],accept_multiple_files=True)
build_btn=st.sidebar.button("🚀 Build Vector Store")
clear_btn=st.sidebar.button("🧹 Reset")

if "vs" not in st.session_state: st.session_state.vs=None
if "dfs" not in st.session_state: st.session_state.dfs={}

if clear_btn:
    st.session_state.vs=None; st.session_state.dfs={}
    st.success("Reset done")

if build_btn and uploaded:
    with st.spinner("Building vector store..."):
        docs=build_documents_from_uploads(uploaded)
        st.session_state.vs=build_faiss(docs)
        st.success(f"Indexed {len(docs)} chunks")

if st.session_state.dfs:
    st.subheader("📊 Data Analysis")
    for fname,payload in st.session_state.dfs.items():
        with st.expander(fname):
            for sheet,df in payload["sheets"].items():
                st.markdown(f"**Sheet: {sheet}**")
                auto_analyze_dataframe(df,fname,sheet)

# Query
st.subheader("💬 Pertanyaan")
q=st.text_input("Tanya berdasarkan dokumen:")
if st.button("Tanyakan"):
    if not q.strip():
        st.warning("Isi pertanyaan dulu")
    elif not st.session_state.vs:
        st.info("Vector store belum ada")
    else:
        res=st.session_state.vs.similarity_search(q,k=5)
        ctx="\n".join([d.page_content for d in res])
        prompt=f"Jawab berdasarkan konteks:\n{ctx}\n\nPertanyaan:{q}\nJawaban:"
        try:
            if st.sidebar.radio("LLM:",["Gemini","Groq"],horizontal=True)=="Gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.2)
                ans=llm.invoke(prompt)
            else:
                from langchain_groq import ChatGroq
                llm=ChatGroq(model_name="llama-3.3-70b-versatile",groq_api_key=GROQ_API_KEY,temperature=0.2)
                ans=llm.invoke(prompt)
            st.write(getattr(ans,"content",str(ans)))
        except Exception as e:
            st.error(str(e))
