from pathlib import Path
from typing import List

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

from .utils import normalize_text

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def _annotate_docs(docs: List[Document], file_path: Path) -> List[Document]:
    out = []
    for doc in docs:
        text = normalize_text(doc.page_content)
        if not text:
            continue

        meta = dict(doc.metadata or {})
        meta["source_path"] = str(file_path)
        meta["file_name"] = file_path.name
        meta["file_type"] = file_path.suffix.lower().lstrip(".")
        doc.page_content = text
        doc.metadata = meta
        out.append(doc)

    return out


def _load_text_file(file_path: Path) -> List[Document]:
    try:
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
    except Exception:
        loader = TextLoader(str(file_path), encoding="latin-1")
        docs = loader.load()

    return _annotate_docs(docs, file_path)


def load_documents_from_folder(folder_path: Path) -> List[Document]:
    if not folder_path.exists():
        return []

    all_docs: List[Document] = []

    for file_path in sorted(folder_path.rglob("*")):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        try:
            if ext == ".pdf":
                docs = PyPDFLoader(str(file_path)).load()
            elif ext == ".docx":
                docs = Docx2txtLoader(str(file_path)).load()
            else:
                docs = _load_text_file(file_path)

            all_docs.extend(_annotate_docs(docs, file_path))
        except Exception as e:
            print(f"Skipping {file_path.name}: {e}")

    return all_docs
