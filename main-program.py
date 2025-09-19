import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
    UnstructuredImageLoader,
)
import tempfile, os

        
# =====================
# Init Session State
# =====================
if "dfs" not in st.session_state:
    st.session_state.dfs = {}
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ======================
# LLM SETUP (Gemini + Groq fallback)
# ======================
def load_llm():
    try:
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    except Exception:
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

llm = load_llm()

# ======================
# Helpers  Utility Functions
# ======================
def df_info_text(df: pd.DataFrame) -> str:
    """Ringkasan dataset sederhana"""
    info = f"Baris: {df.shape[0]}, Kolom: {df.shape[1]}\n"
    info += "Kolom:\n" + ", ".join(df.columns[:30])
    if df.shape[1] > 30:
        info += " ..."
    return info
    
def detect_data_types(df):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    return cat_cols, num_cols

def safe_describe(df):
    try:
        return df.describe(include="all")
    except Exception:
        return pd.DataFrame()

def detect_column_types(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    return numeric_cols, categorical_cols, datetime_cols

def generate_dataset_insight(df: pd.DataFrame):
    stats = safe_describe(df).reset_index().to_string()
    prompt = ChatPromptTemplate.from_template("""
    Kamu adalah analis data. 
    Berdasarkan statistik berikut:
    {stats}
    
    Buatkan insight utama dan kesimpulan secara akurat, jelas dan sedetil mungkin berdasarkan konteks dengan bahasa natural, sopan, ringkas, dan mudah dipahami.
    Jika konteks berupa ringkasan tabel (CSV/Excel), gunakan metrik yang tersedia (shape, dtypes, missing, describe, sample). 
    Jika jawaban tidak ada, katakan: "Jawaban tidak tersedia dalam konteks yang diberikan"
    """)
    chain = prompt | llm
    return chain.invoke({"stats": stats}).content

def load_document(file_path, file_type):
    if file_type == ".pdf":
        return PyPDFLoader(file_path).load()
    elif file_type == ".txt":
        return TextLoader(file_path).load()
    elif file_type == ".docx":
        return Docx2txtLoader(file_path).load()
    elif file_type in [".pptx", ".ppt"]:
        return UnstructuredPowerPointLoader(file_path).load()
    elif file_type.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
        return UnstructuredImageLoader(file_path).load()
    else:
        return []

def process_rag_files(uploaded_files):
    docs = []
    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        docs.extend(load_document(tmp_path, suffix))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_split = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs_split, embeddings)

# =====================
# UI Tabs
# =====================
st.set_page_config(page_title="📊 Data & Document Chatbot", layout="wide")

tab1, tab2 = st.tabs(["📈 Data Analysis", "📚 RAG Advanced"])

# ====== MODE 1: Data Analysis ======
with tab1:
    uploaded_file = st.file_uploader("Upload file Excel/CSV untuk analisa data", type=["csv", "xls", "xlsx"])
    df = None

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            xls = pd.ExcelFile(uploaded_file)
            st.session_state.dfs = {sheet: pd.read_excel(uploaded_file, sheet_name=sheet) for sheet in xls.sheet_names}
            selected_sheets = st.multiselect("📑 Pilih Sheet", list(st.session_state.dfs.keys()), default=list(st.session_state.dfs.keys())[:1])
            if selected_sheets:
                df_list = []
                for s in selected_sheets:
                    temp = st.session_state.dfs[s].copy()
                    temp["SheetName"] = s
                    df_list.append(temp)
                df = pd.concat(df_list, ignore_index=True)

    if df is not None:
        st.dataframe(df.head(10))
        
        # deteksi tipe kolom
        numeric_cols, categorical_cols, datetime_cols = detect_column_types(df)
        st.write(f"Kolom Numerik: {numeric_cols}")
        st.write(f"Kolom Kategorikal: {categorical_cols}")
        st.write(f"Kolom Datetime: {datetime_cols}")

        st.text(df_info_text(df))
        st.write(f"**Data shape:** {df.shape}")        

        # dropdown axis
        st.subheader("⚙️ Pilih Kolom untuk Visualisasi")
        x_axis = st.selectbox("Kolom X Axis", df.columns)
        y_axis = st.selectbox("Kolom Y Axis", df.columns)

        if x_axis and y_axis:
            x_is_num = x_axis in numeric_cols or x_axis in datetime_cols
            y_is_num = y_axis in numeric_cols
            x_is_cat = x_axis in categorical_cols
            y_is_cat = y_axis in categorical_cols

            fig = None
            if x_is_num and y_is_num:
                fig = px.scatter(df, x=x_axis, y=y_axis, title=f"📈 Scatter {y_axis} vs {x_axis}")
            elif x_axis in datetime_cols and y_is_num:
                fig = px.line(df, x=x_axis, y=y_axis, title=f"📈 Tren {y_axis} vs {x_axis}")
            elif x_is_cat and y_is_num:
                fig = px.bar(df, x=x_axis, y=y_axis, title=f"📊 Bar {y_axis} per {x_axis}")
            elif x_is_num and y_is_cat:
                fig = px.bar(df, x=y_axis, y=x_axis, title=f"📊 Bar {x_axis} per {y_axis}")
            elif x_is_cat and y_is_cat:
                crosstab = pd.crosstab(df[x_axis], df[y_axis])
                fig = px.imshow(crosstab, title=f"🔢 Frekuensi {x_axis} vs {y_axis}")

            if fig:
                st.plotly_chart(fig, use_container_width=True)

        # chatbot untuk insight
        st.subheader("💬 Chatbot Data Analysis")
        q = st.text_input("Tanyakan sesuatu tentang dataset")
        if q:
            if "insight" in q.lower() or "kesimpulan" in q.lower():
                st.write(generate_dataset_insight(df))
            else:
                st.write("🔍 Gunakan kata kunci statistik / tren / kategori / insight untuk analisis.")

# ====== MODE 2: RAG Advanced ======
with tab2:
    uploaded_files = st.file_uploader("Upload dokumen (PDF, DOCX, PPTX, TXT, JPG, PNG, dsb)", type=["pdf","docx","pptx","txt","jpg","jpeg","png","bmp","gif"], accept_multiple_files=True)
    if uploaded_files:
        st.session_state.vectorstore = process_rag_files(uploaded_files)
        st.success("✅ Dokumen berhasil diproses!")

    st.subheader("💬 Chatbot RAG")
    q2 = st.text_input("Tanyakan sesuatu tentang dokumen")
    if q2 and st.session_state.vectorstore:
        retriever = st.session_state.vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(q2)
        context = "\n".join([d.page_content for d in docs[:3]])

        prompt = ChatPromptTemplate.from_template("""
        Jawab pertanyaan berikut secara akurat, jelas dan sedetil mungkin berdasarkan dokumen konteks dengan bahasa natural, sopan, ringkas, dan mudah dipahami.
        Jika konteks berupa ringkasan tabel (CSV/Excel), gunakan metrik yang tersedia (shape, dtypes, missing, describe, sample). 
        Jika jawaban tidak ada, katakan: "Jawaban tidak tersedia dalam konteks yang diberikan"
        Pertanyaan: {q}
        Konteks: {context}
        Jawaban ringkas:
        """)
        chain = prompt | llm
        st.write(chain.invoke({"q": q2, "context": context}).content)
