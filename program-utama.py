import os
from io import BytesIO
import requests
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

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
# Session state init (fix bug)
# -------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "dataframes" not in st.session_state:
    st.session_state.dataframes = {}  # { filename: { "sheets": {sheet_name: df, ...} }, ... }

# -------------------------
# Sidebar: Upload + LLM choice + Actions
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
# Helpers / compatibility
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
        resp = requests.post(
            "https://api.ocr.space/parse/image",
            files={"file": (filename, file_bytes, "image/png")},
            data={"apikey": OCR_SPACE_API_KEY, "language": "eng"},
            timeout=60
        )
        result = resp.json()
        if result.get("IsErroredOnProcessing"):
            st.warning("⚠️ OCR.Space error: " + str(result.get("ErrorMessage", ["Unknown error"])))
            return ""
        parsed = []
        for p in result.get("ParsedResults", []):
            parsed.append(p.get("ParsedText", ""))
        return "\n".join(parsed).strip()
    except Exception as e:
        st.warning(f"⚠️ OCR error: {e}")
        return ""

# CSV / Excel extractors that also store DataFrames in session_state.dataframes
def extract_text_from_csv(file_bytes, filename):
    try:
        file_bytes.seek(0)
        # try read default encoding; fallback latin-1
        try:
            df = pd.read_csv(file_bytes)
        except Exception:
            file_bytes.seek(0)
            df = pd.read_csv(file_bytes, encoding="latin-1")
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
# Auto-analysis UI per DataFrame
# -------------------------
def auto_analyze_dataframe(df, filename, sheet_name, show_in_app=True):
    """
    Displays:
      - Head(10), Tail(10)
      - describe()
      - info()
      - correlation (numeric)
      - trend (if datetime column found)
      - outliers (IQR)
      - flexible Top/Bottom-N (select column, N, order)
      - download Excel & HTML reports
    """
    rows, cols = df.shape

    # Prepare textual report
    parts = []
    parts.append(f"Dataset: {filename} — sheet: {sheet_name}")
    parts.append(f"Shape: {rows} rows × {cols} cols")
    parts.append("Dtypes:\n" + df.dtypes.astype(str).to_string())
    parts.append("Missing counts:\n" + df.isna().sum().to_string())

    # Correlation
    num_df = df.select_dtypes(include="number")
    if not num_df.empty:
        parts.append("Correlation matrix:\n" + num_df.corr().to_string())

    # Trend (find first date-like column)
    date_trend_text = ""
    for col in df.columns:
        if any(k in col.lower() for k in ["date", "time", "tanggal"]):
            try:
                df[col] = pd.to_datetime(df[col])
                trend = df.groupby(df[col].dt.to_period("M")).size()
                date_trend_text = f"Trend by {col}:\n{trend.to_string()}"
                parts.append(date_trend_text)
                break
            except Exception:
                pass

    # Outliers (IQR) for numeric columns
    outlier_texts = []
    if not num_df.empty:
        for c in num_df.columns:
            q1 = num_df[c].quantile(0.25)
            q3 = num_df[c].quantile(0.75)
            iqr = q3 - q1
            out = num_df[(num_df[c] < q1 - 1.5 * iqr) | (num_df[c] > q3 + 1.5 * iqr)]
            if not out.empty:
                outlier_texts.append(f"Outliers for {c}: {len(out)} rows (showing head 20)\n{out.head(20).to_string()}")
    if outlier_texts:
        parts.extend(outlier_texts)

    report_text = "\n\n".join(parts)

    # Excel export with multiple sheets
    out_excel = BytesIO()
    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        try:
            df.to_excel(writer, sheet_name="Data", index=False)
        except Exception:
            df.reset_index().to_excel(writer, sheet_name="Data", index=False)
        # write describe & correlation if available
        try:
            safe_describe(df).to_excel(writer, sheet_name="Describe")
        except Exception:
            pass
        if not num_df.empty:
            num_df.corr().to_excel(writer, sheet_name="Correlation")
    out_excel.seek(0)

    # HTML export (simple, readable)
    html_body = f"<h2>Analysis — {filename} / {sheet_name}</h2>"
    html_body += f"<pre>{report_text}</pre>"
    html_bytes = html_body.encode("utf-8")

    # Display in app
    if show_in_app:
        st.markdown(f"### 📄 Analisa: {filename} — {sheet_name}")
        # Head / Tail
        st.write("**Head (10):**")
        st.dataframe(df.head(10))
        st.write("**Tail (10):**")
        st.dataframe(df.tail(10))

        # describe()
        st.write("**describe():**")
        try:
            st.dataframe(safe_describe(df))
        except Exception as e:
            st.write("describe() error:", e)

        # info()
        st.write("**info():**")
        st.text(df_info_text(df))

        # show correlation heatmap if numeric
        if not num_df.empty:
            try:
                corr = num_df.corr()
                st.write("**Correlation matrix:**")
                st.dataframe(corr)
                fig, ax = plt.subplots()
                cax = ax.matshow(corr)
                fig.colorbar(cax)
                ax.set_xticks(range(len(corr.columns)))
                ax.set_xticklabels(corr.columns, rotation=90)
                ax.set_yticks(range(len(corr.columns)))
                ax.set_yticklabels(corr.columns)
                st.pyplot(fig)
            except Exception:
                pass

        # automatic Top10 for Sales/Profit if present (backward compat)
        for cand in ["sales", "profit"]:
            for col in df.columns:
                if col.lower() == cand:
                    try:
                        st.markdown(f"**Top 10 berdasarkan {col} (descending):**")
                        st.dataframe(df.sort_values(by=col, ascending=False).head(10))
                    except Exception:
                        pass

        # Flexible Top/Bottom-N UI
        st.markdown("---")
        st.markdown("### Pilih kolom untuk Top/Bottom-N (flexible)")
        col_list = list(df.columns)
        if col_list:
            widget_prefix = f"{filename}___{sheet_name}"
            chosen_col = st.selectbox("Pilih kolom:", options=col_list, key=f"col_{widget_prefix}")
            max_n = min(100, max(1, len(df)))
            chosen_n = st.slider("Jumlah baris (N):", min_value=1, max_value=max_n, value=min(10, max_n), key=f"n_{widget_prefix}")
            sort_order = st.radio("Urutan:", ["Top N (descending)", "Bottom N (ascending)"], index=0, key=f"ord_{widget_prefix}", horizontal=True)
            if st.button("Tampilkan Top/Bottom N", key=f"btn_{widget_prefix}"):
                try:
                    if pd.api.types.is_numeric_dtype(df[chosen_col]):
                        asc = (sort_order == "Bottom N (ascending)")
                        topn = df.sort_values(by=chosen_col, ascending=asc).head(chosen_n)
                        st.markdown(f"**{sort_order} berdasarkan {chosen_col}:**")
                        st.dataframe(topn)
                    else:
                        # for non-numeric show top value_counts or bottom (least frequent)
                        if sort_order == "Top N (descending)":
                            vc = df[chosen_col].value_counts().head(chosen_n)
                        else:
                            vc = df[chosen_col].value_counts(ascending=True).head(chosen_n)
                        st.markdown(f"**{sort_order} nilai kolom {chosen_col}:**")
                        st.dataframe(vc.to_frame(name="count"))
                        st.markdown("Sample rows for these values:")
                        sample_vals = vc.index.tolist()
                        st.dataframe(df[df[chosen_col].isin(sample_vals)].head(200))
                except Exception as e:
                    st.warning(f"⚠️ Gagal menampilkan Top/Bottom-N: {e}")

    # Download buttons
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            label="⬇️ Download laporan Excel",
            data=out_excel,
            file_name=f"analysis_{filename}_{sheet_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    with c2:
        st.download_button(
            label="⬇️ Download laporan HTML",
            data=html_bytes,
            file_name=f"analysis_{filename}_{sheet_name}.html",
            mime="text/html"
        )

# -------------------------
# Handle upload parsing (store DataFrames & indexable text)
# -------------------------
st.title("🤖 Multi-file Chatbot + Auto Analysis (final)")

if uploaded_files:
    st.sidebar.info(f"{len(uploaded_files)} file dipilih — klik 'Build Vector Store' untuk indexing")
    # parse uploaded files immediately to fill session_state.dataframes (but do not build vectors until button)
    for f in uploaded_files:
        # only parse if not already parsed (prevents duplicate parsing)
        if f.name not in st.session_state.dataframes:
            try:
                extract_text_from_file(f)
                st.sidebar.success(f"Parsed: {f.name}")
            except Exception as e:
                st.sidebar.warning(f"Failed parse {f.name}: {e}")

# Build vector store action
if build_btn:
    if not uploaded_files:
        st.sidebar.warning("Silakan upload minimal 1 file sebelum membangun vector store.")
    else:
        with st.spinner("📦 Memproses file dan membangun vector store..."):
            docs = build_documents_from_uploads(uploaded_files)
            if not docs:
                st.sidebar.error("Tidak ada teks valid berhasil diekstrak. Periksa file.")
            else:
                vs = build_faiss_from_documents(docs)
                st.session_state.vector_store = vs
                st.session_state.indexed_files = [f.name for f in uploaded_files]
                st.sidebar.success(f"Vector store terbangun. Dokumen terindeks: {len(st.session_state.indexed_files)} | Chunks: {len(docs)}")

# Reset action
if clear_btn:
    st.session_state.vector_store = None
    st.session_state.indexed_files = []
    st.session_state.dataframes = {}
    st.sidebar.success("Session state direset.")

# Show indexed files or hint
if st.session_state.indexed_files:
    st.markdown("**Dokumen terindeks:**")
    st.write(" • " + "\n • ".join(st.session_state.indexed_files))
else:
    st.info("Belum ada dokumen terindeks. Upload file lalu klik 'Build Vector Store' bila ingin pakai fitur tanya jawab LLM.")

# Data preview & analysis area
if st.session_state.dataframes:
    st.subheader("📊 Data Preview, Profiling & Analisa Otomatis")
    show_in_app = st.checkbox("Tampilkan analisa di Streamlit (per sheet)", value=True)
    for fname, payload in st.session_state.dataframes.items():
        with st.expander(f"🔎 File: {fname}", expanded=False):
            for sheet_name, df in payload["sheets"].items():
                auto_analyze_dataframe(df, fname, sheet_name, show_in_app)

# -------------------------
# Query area (always visible)
# -------------------------
st.subheader("💬 Ajukan Pertanyaan")
prompt = st.text_input("Tanyakan sesuatu berdasarkan dokumen/tabel yang diupload:",
                       placeholder="Misal: Ringkas tren penjualan per wilayah 2024")
ask_btn = st.button("Tanyakan")

if ask_btn:
    if not prompt.strip():
        st.warning("Masukkan pertanyaan terlebih dahulu.")
    elif st.session_state.vector_store is None:
        st.info("Belum ada vector store. Upload file dan klik 'Build Vector Store' jika ingin jawaban berbasis dokumen.")
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
