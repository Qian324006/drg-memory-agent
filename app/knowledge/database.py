from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os

from app.config import ENV_FILE, KNOWLEDGE_DB_DIR

load_dotenv(ENV_FILE, override=True)

embeddings = OpenAIEmbeddings(
    model="text-embedding-v3",
    api_key=os.getenv("PROXY_API_KEY"),
    base_url=os.getenv("PROXY_BASE_URL"),
    check_embedding_ctx_length=False,
    max_retries=3,
)

knowledge_db = Chroma(
    collection_name="medical_drg_knowledge",
    embedding_function=embeddings,
    persist_directory=str(KNOWLEDGE_DB_DIR),
    collection_metadata={"hnsw:space": "cosine"}
)
