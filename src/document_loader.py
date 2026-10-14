"""Document loading and context-aware chunking for the prototype."""
from pathlib import Path
import re

def load_txt(path):
    path = Path(path)
    return [{"document_id": path.stem, "source": path.name, "page": 1, "text": path.read_text(encoding="utf-8", errors="ignore")}]

def load_pdf(path):
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required to read PDFs. Install requirements.txt.") from exc
    rows = []
    with fitz.open(path) as doc:
        for idx, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                rows.append({
                    "document_id": Path(path).stem,
                    "source": Path(path).name,
                    "page": idx,
                    "text": text
                })
    return rows

def load_file(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in {".txt", ".md"}:
        return load_txt(path)
    raise ValueError(f"Unsupported file type: {suffix}. Use PDF, TXT, or MD.")

def load_sample_documents(folder="data/sample_documents"):
    folder = Path(folder)
    pages = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in {".txt", ".md", ".pdf"}:
            pages.extend(load_file(path))
    return pages

def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_pages(pages, max_chars=700, overlap=100):
    """Split by paragraphs, then enforce a character limit. Preserve source metadata."""
    chunks = []
    for page in pages:
        text = clean_text(page.get("text", ""))
        if not text:
            continue
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        buffer = ""
        for para in paragraphs:
            if len(para) > max_chars:
                if buffer:
                    chunks.append(_chunk_row(page, buffer))
                    buffer = ""
                start = 0
                while start < len(para):
                    piece = para[start:start + max_chars]
                    chunks.append(_chunk_row(page, piece))
                    if start + max_chars >= len(para):
                        break
                    start += max_chars - overlap
                continue
            candidate = f"{buffer}\n{para}".strip() if buffer else para
            if len(candidate) <= max_chars:
                buffer = candidate
            else:
                chunks.append(_chunk_row(page, buffer))
                carry = buffer[-overlap:] if overlap else ""
                buffer = f"{carry}\n{para}".strip()
        if buffer:
            chunks.append(_chunk_row(page, buffer))
    for i, chunk in enumerate(chunks):
        chunk["chunk_id"] = f'{chunk["document_id"]}-C{i+1:03d}'
    return chunks

def _chunk_row(page, text):
    return {
        "document_id": page.get("document_id", "unknown"),
        "source": page.get("source", "unknown"),
        "page": page.get("page", 1),
        "text": text.strip()
    }
