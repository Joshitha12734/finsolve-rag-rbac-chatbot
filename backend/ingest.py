"""
Loads all source documents under app/data/<department>/ and splits them
into retrieval-sized chunks, each tagged with metadata:
    - department: engineering | finance | marketing | hr | general
    - source: relative file path (used for citations)
    - chunk_id: stable id for the chunk

Supports .md/.txt (paragraph-aware splitting) and .csv (row-wise,
turned into natural-language sentences so they embed/match well).
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from .config import DATA_DIR

CHUNK_TARGET_CHARS = 900
CHUNK_OVERLAP_CHARS = 150

CLASSIFICATION_MANIFEST_PATH = DATA_DIR / "_classification.json"
DEFAULT_CLASSIFICATION = "internal"


def _load_classification_manifest() -> dict:
    if not CLASSIFICATION_MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(CLASSIFICATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@dataclass
class Chunk:
    chunk_id: str
    department: str
    source: str
    text: str
    classification: str = DEFAULT_CLASSIFICATION


def _split_markdown(text: str) -> List[str]:
    """Paragraph-aware splitter that groups paragraphs up to a target size,
    with a small overlap between consecutive chunks for context continuity."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= CHUNK_TARGET_CHARS:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
                # carry a small overlap tail forward for continuity
                overlap = current[-CHUNK_OVERLAP_CHARS:]
                current = f"{overlap}\n\n{para}".strip()
            else:
                # single paragraph longer than target: hard split
                for i in range(0, len(para), CHUNK_TARGET_CHARS):
                    chunks.append(para[i:i + CHUNK_TARGET_CHARS])
                current = ""
    if current:
        chunks.append(current)
    return chunks


def _load_markdown_file(path: Path, department: str, classification: str) -> List[Chunk]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    rel = str(path.relative_to(DATA_DIR))
    pieces = _split_markdown(text)
    return [
        Chunk(chunk_id=f"{rel}::{i}", department=department, source=rel, text=piece, classification=classification)
        for i, piece in enumerate(pieces)
    ]


def _load_csv_file(path: Path, department: str, classification: str) -> List[Chunk]:
    """Turn each CSV row into a natural-language sentence so it can be
    matched against free-text queries, e.g. HR employee records."""
    rel = str(path.relative_to(DATA_DIR))
    chunks: List[Chunk] = []
    with path.open(newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            sentence = "; ".join(f"{k}: {v}" for k, v in row.items() if v not in (None, ""))
            chunks.append(Chunk(chunk_id=f"{rel}::{i}", department=department, source=rel, text=sentence, classification=classification))
    return chunks


def load_all_chunks() -> List[Chunk]:
    manifest = _load_classification_manifest()
    chunks: List[Chunk] = []
    for dept_dir in sorted(DATA_DIR.iterdir()):
        if not dept_dir.is_dir():
            continue
        department = dept_dir.name
        for file_path in sorted(dept_dir.glob("**/*")):
            if not file_path.is_file():
                continue
            rel = str(file_path.relative_to(DATA_DIR))
            classification = manifest.get(rel, DEFAULT_CLASSIFICATION)
            if file_path.suffix.lower() in {".md", ".txt"}:
                chunks.extend(_load_markdown_file(file_path, department, classification))
            elif file_path.suffix.lower() == ".csv":
                chunks.extend(_load_csv_file(file_path, department, classification))
    return chunks


def chunks_to_dicts(chunks: List[Chunk]) -> List[dict]:
    return [asdict(c) for c in chunks]


if __name__ == "__main__":
    all_chunks = load_all_chunks()
    print(f"Loaded {len(all_chunks)} chunks from {DATA_DIR}")
    by_dept = {}
    for c in all_chunks:
        by_dept[c.department] = by_dept.get(c.department, 0) + 1
    for dept, count in by_dept.items():
        print(f"  {dept}: {count} chunks")
