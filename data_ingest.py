# data_ingest.py (concept)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
import chromadb

loader = PyPDFLoader("manual.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
chunks = splitter.split_documents(docs)

# embeddings
emb = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# create chroma client and upsert chunks (title, text)
