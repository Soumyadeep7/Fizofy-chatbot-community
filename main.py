# pyright: ignore[reportMissingImports]
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from uuid import uuid4
from dotenv import load_dotenv
import os

# ----------------- Load .env -----------------
load_dotenv("wwe.env")  # Make sure GOOGLE_API_KEY is in this file

# ----------------- Configuration -----------------
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# ----------------- Embeddings -----------------
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="text-embedding-gecko-001",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# ----------------- Vector Store -----------------
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# ----------------- Load PDFs -----------------
loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()

# ----------------- Split Documents -----------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_documents(raw_documents)

# ----------------- Assign Unique IDs -----------------
uuids = [str(uuid4()) for _ in range(len(chunks))]

# ----------------- Add Chunks to Vector Store -----------------
vector_store.add_documents(documents=chunks, ids=uuids)

print(f"Embedded {len(chunks)} chunks from PDFs in {DATA_PATH}")
