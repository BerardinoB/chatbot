import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Make sure your key is available (export OPENAI_API_KEY=... before running)
if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("Set OPENAI_API_KEY in your environment before running this script.")

# 1) Load your fixed context file
loader = TextLoader("berardino_context.txt", encoding="utf-8")
docs = loader.load()

# 2) Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3) Create embeddings (choose explicit model to be safe)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 4) Build FAISS index
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5) Save locally in a folder called "faiss_index"
vectorstore.save_local("faiss_index")

print("FAISS index saved to ./faiss_index")
