# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def load_pdf_pages(pdf_path: str) -> List[Document]:
    """
    Carrega um PDF em uma lista de Document, 1 por página,
    garantindo metadados consistentes (page 0-based, source, file_path).
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    for i, d in enumerate(pages):
        d.metadata = dict(d.metadata or {})
        d.metadata["page"] = i              # 0-based
        d.metadata.setdefault("source", Path(pdf_path).name)
        d.metadata["file_path"] = str(Path(pdf_path).resolve())

    return pages

