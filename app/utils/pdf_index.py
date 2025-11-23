# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple, List

from langchain_core.documents import Document

from .rag import indexar_pdf as _indexar_pdf_rag


def indexar_pdf(pdf_path: str) -> Tuple[object, int, int, List[Document]]:
    """
    Fachada para a camada de RAG modular.

    Retorna:
      - retriever (objeto com .invoke ou .as_retriever)
      - n_pages
      - n_chunks
      - docs_paginas (lista Document por página)
    """
    return _indexar_pdf_rag(pdf_path)

