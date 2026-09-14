import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def snippet(text: str, length: int = 260) -> str:
    cleaned = normalize_text(text)
    if len(cleaned) <= length:
        return cleaned
    return cleaned[:length].rstrip() + "..."


def source_label(metadata: Dict[str, Any]) -> str:
    file_name = metadata.get("file_name") or Path(metadata.get("source_path", "unknown")).name
    page = metadata.get("page", None)

    if isinstance(page, int):
        return f"{file_name} (p. {page + 1})"
    return file_name


def chunk_key(metadata: Dict[str, Any]) -> str:
    file_name = metadata.get("file_name") or Path(metadata.get("source_path", "unknown")).name
    page = metadata.get("page", "na")
    start_index = metadata.get("start_index", metadata.get("chunk_index", "na"))
    return f"{file_name}::p{page}::s{start_index}"


def unique_by_key(items: List[Dict[str, Any]], key_field: str = "key") -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for item in items:
        key = item.get(key_field)
        if key not in seen:
            out.append(item)
            seen.add(key)
    return out
