"""
document_reader.py — Lecteur et formateur de documents pour Neural Forge.
"""

import json
import os
from enum import Enum

DIRECT_THRESHOLD = 12_000   # ~3 000 tokens
CHUNK_SIZE = 8_000          # ~2 000 tokens par chunk
MAX_INJECT_CHARS = 20_000


class InjectionStrategy(Enum):
    DIRECT = "direct"
    SUMMARY = "summary"


class DocumentReadError(Exception):
    pass


class DocumentResult:
    """Résultat de la lecture d'un document prêt à être injecté."""

    def __init__(
        self,
        content: str,
        strategy: InjectionStrategy,
        original_chars: int,
        filename: str,
    ):
        self.content = content
        self.strategy = strategy
        self.original_chars = original_chars
        self.filename = filename

    @property
    def was_summarized(self) -> bool:
        return self.strategy == InjectionStrategy.SUMMARY

    def build_prompt(self, user_question: str = "") -> str:
        if self.was_summarized:
            header = (
                f"Voici un résumé automatique du document « {self.filename} » "
                f"(document original : {self.original_chars:,} caractères) :\n\n"
                f"{self.content}"
            )
        else:
            header = (
                f"Voici le contenu du document « {self.filename} » :\n\n"
                f"{self.content}"
            )

        if user_question:
            return f"{header}\n\n---\nQuestion : {user_question}"
        return f"{header}\n\n---\nFais un résumé clair et structuré de ce document."


def read_document(
    path: str,
    llm_node=None,
    direct_threshold: int = DIRECT_THRESHOLD,
) -> DocumentResult:
    if not path or not os.path.isfile(path):
        raise DocumentReadError(f"Fichier introuvable : {path}")

    filename = os.path.basename(path)
    raw = _extract_text(path)
    original_chars = len(raw)

    if original_chars <= direct_threshold:
        return DocumentResult(
            content=raw,
            strategy=InjectionStrategy.DIRECT,
            original_chars=original_chars,
            filename=filename,
        )

    if llm_node is not None and getattr(llm_node, "model_loaded", False):
        summary = _summarize_in_chunks(raw, llm_node, filename)
        return DocumentResult(
            content=summary,
            strategy=InjectionStrategy.SUMMARY,
            original_chars=original_chars,
            filename=filename,
        )

    truncated = _truncate(raw, MAX_INJECT_CHARS)
    note = (
        f"\n\n[⚠ Document tronqué : {original_chars:,} → {MAX_INJECT_CHARS:,} car. "
        "Chargez un modèle pour activer le résumé automatique.]"
    )
    return DocumentResult(
        content=truncated + note,
        strategy=InjectionStrategy.DIRECT,
        original_chars=original_chars,
        filename=filename,
    )


def _run_stateless_llm(llm_node, prompt: str) -> str:
    """Exécute une requête interne sans polluer conversation_history ni ChromaDB."""
    if getattr(llm_node, "llm_engine", None) is not None:
        output = llm_node.llm_engine.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.2,
            stream=False,
        )
        return output["choices"][0]["message"]["content"].strip()

    # Fallback compatible avec les mocks de tests unitaires (FakeLLMNode)
    from core.types import DataPacket, DataType
    saved_hist = list(getattr(llm_node, "conversation_history", []))
    try:
        llm_node.set_input(DataPacket(DataType.TEXT, prompt, source="document_reader"))
        res = llm_node._run_inference()
        return res.content.strip()
    finally:
        if hasattr(llm_node, "conversation_history"):
            llm_node.conversation_history = saved_hist


def _summarize_in_chunks(text: str, llm_node, filename: str) -> str:
    chunks = _split_chunks(text, CHUNK_SIZE)
    partial_summaries = []

    for i, chunk in enumerate(chunks):
        prompt = (
            f"Résume ce passage ({i + 1}/{len(chunks)}) "
            f"du document « {filename} » en 3-5 phrases concises :\n\n{chunk}"
        )
        try:
            summary_part = _run_stateless_llm(llm_node, prompt)
            partial_summaries.append(f"[Partie {i + 1}]\n{summary_part}")
        except Exception:
            partial_summaries.append(f"[Partie {i + 1}]\n{_truncate(chunk, 500)}")

    combined = "\n\n".join(partial_summaries)

    if len(combined) > DIRECT_THRESHOLD and len(chunks) > 1:
        final_prompt = (
            f"Voici les résumés partiels du document « {filename} ». "
            "Synthétise-les en un résumé global cohérent et structuré :\n\n"
            f"{combined}"
        )
        try:
            return _run_stateless_llm(llm_node, final_prompt)
        except Exception:
            pass

    return combined


def _extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md", ".rst", ".csv", ".log"):
        return _read_text(path)
    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".docx":
        return _read_docx(path)
    if ext in (".json", ".jsonl"):
        return _read_json(path)
    return _read_text(path)


def _read_image_fast(path: str) -> str:
    """Extraction instantanée via OCR classique (contourne LLaVA)."""
    try:
        import pytesseract
        from PIL import Image

        with Image.open(path) as img:
            text = pytesseract.image_to_string(img, lang="fra+eng").strip()

        if not text:
            raise DocumentReadError(
                "L'OCR rapide n'a détecté aucun texte. "
                "Le fichier nécessite une analyse visuelle profonde (LLaVA)."
            )
        return text
    except ImportError as exc:
        raise DocumentReadError(
            "Le module 'pytesseract' n'est pas installé. Tapez : pip install pytesseract"
        ) from exc
    except DocumentReadError:
        raise
    except Exception as exc:
        raise DocumentReadError(f"Échec de l'OCR rapide : {exc}") from exc


def _read_text(path: str) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise DocumentReadError(f"Encodage inconnu : '{path}'")


def _read_pdf(path: str) -> str:
    try:
        import fitz
    except ImportError as exc:
        raise DocumentReadError("PyMuPDF non installé — lancez : pip install pymupdf") from exc

    try:
        pages = []
        with fitz.open(path) as doc:
            for i, page in enumerate(doc):
                text = page.get_text("text").strip()
                if text:
                    pages.append(f"[Page {i + 1}]\n{text}")
        if not pages:
            raise DocumentReadError("Aucun texte extractible (PDF scanné ou protégé).")
        return "\n\n".join(pages)
    except DocumentReadError:
        raise
    except Exception as exc:
        raise DocumentReadError(f"Erreur PDF : {exc}") from exc


def _read_docx(path: str) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise DocumentReadError("python-docx non installé — lancez : pip install python-docx") from exc

    try:
        doc = Document(path)
        parts = [para.text for para in doc.paragraphs if para.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return "\n\n".join(parts)
    except Exception as exc:
        raise DocumentReadError(f"Erreur DOCX : {exc}") from exc


def _read_json(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".jsonl":
            lines = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        lines.append(json.dumps(obj, ensure_ascii=False, indent=2))
                    except json.JSONDecodeError:
                        lines.append(line)
            return "\n---\n".join(lines)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as exc:
        raise DocumentReadError(f"Erreur JSON : {exc}") from exc


def _split_chunks(text: str, chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        cut = text.rfind("\n\n", start, end)
        if cut <= start:
            cut = text.rfind("\n", start, end)
        if cut <= start:
            cut = end
        chunks.append(text[start:cut])
        start = cut
    return [c.strip() for c in chunks if c.strip()]


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_nl = truncated.rfind("\n")
    if last_nl > max_chars * 0.85:
        truncated = truncated[:last_nl]
    return truncated


def get_file_info(path: str) -> dict:
    size_kb = os.path.getsize(path) / 1024 if path and os.path.exists(path) else 0.0
    ext = os.path.splitext(path)[1].lower() if path else ""
    label = {
        ".pdf": "PDF", ".docx": "Word", ".txt": "Texte",
        ".md": "Markdown", ".json": "JSON", ".jsonl": "JSONL",
    }.get(ext, "Fichier")
    return {
        "name": os.path.basename(path) if path else "Inconnu",
        "label": label,
        "size_kb": round(size_kb, 1),
        "ext": ext,
    }