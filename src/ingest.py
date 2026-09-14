import argparse
from pathlib import Path
from typing import Dict, List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    PROCESSED_CHUNKS_FILE,
    RAW_DOCS_DIR,
)
from .loaders import load_documents_from_folder
from .utils import chunk_key, ensure_dir, read_jsonl, write_jsonl


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def prepare_chunks(chunks: List[Document]) -> List[Dict]:
    records = []

    for idx, chunk in enumerate(chunks):
        meta = dict(chunk.metadata or {})
        meta["chunk_index"] = idx
        meta["chunk_id"] = chunk_key(meta)

        records.append(
            {
                "text": chunk.page_content,
                "metadata": meta,
            }
        )

    return records


def build_vectorstore(chunks: List[Document], rebuild: bool = True) -> Chroma:
    ensure_dir(CHROMA_DIR)

    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    # Safer rebuild approach for Windows:
    # delete the collection instead of removing chroma.sqlite3 directly.
    if rebuild:
        try:
            vectorstore.delete_collection()
        except Exception:
            pass

        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )

    ids = []
    for idx, chunk in enumerate(chunks):
        meta = dict(chunk.metadata or {})
        meta["chunk_index"] = idx
        meta["chunk_id"] = chunk_key(meta)
        chunk.metadata = meta
        ids.append(meta["chunk_id"])

    vectorstore.add_documents(chunks, ids=ids)

    return vectorstore


def ingest_corpus(raw_docs_dir: Path = RAW_DOCS_DIR, rebuild: bool = True) -> Dict:
    ensure_dir(raw_docs_dir)
    ensure_dir(CHROMA_DIR)

    documents = load_documents_from_folder(raw_docs_dir)
    if not documents:
        raise ValueError(
            f"No supported documents found in {raw_docs_dir}. "
            "Add PDF, DOCX, TXT, or MD files to proceed."
        )

    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("Document splitting produced no chunks.")

    chunk_records = prepare_chunks(chunks)
    write_jsonl(PROCESSED_CHUNKS_FILE, chunk_records)

    vectorstore = build_vectorstore(chunks, rebuild=rebuild)

    unique_files = sorted({doc.metadata.get("file_name", "unknown") for doc in documents})

    return {
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "files_indexed": unique_files,
        "vectorstore": vectorstore,
    }


def load_indexed_chunks() -> List[Dict]:
    return read_jsonl(PROCESSED_CHUNKS_FILE)


def load_vectorstore() -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )


def index_exists() -> bool:
    return PROCESSED_CHUNKS_FILE.exists() and PROCESSED_CHUNKS_FILE.stat().st_size > 0


def main():
    parser = argparse.ArgumentParser(description="Build the RAG index")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild vector DB from scratch")
    args = parser.parse_args()

    stats = ingest_corpus(rebuild=args.rebuild)
    print("Index built successfully.")
    print(stats)


if __name__ == "__main__":
    main()