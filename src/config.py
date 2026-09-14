from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"
CHROMA_DIR = DATA_DIR / "chroma_db"
PROCESSED_CHUNKS_FILE = DATA_DIR / "processed_chunks.jsonl"

COLLECTION_NAME = "enterprise_knowledge_assistant"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

TOP_K_VECTOR = 6
TOP_K_BM25 = 6
TOP_K_FINAL = 5
RRF_K = 60
