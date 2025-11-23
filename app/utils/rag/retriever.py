# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Tuple, List

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from .loader import load_pdf_pages
from .splitter import split_documents
from .embeddings import get_embeddings
from .vectorstore import build_vectorstore


def indexar_pdf(pdf_path: str) -> Tuple[object, int, int, List[Document]]:
    """
    Orquestra o RAG:

    - Carrega páginas do PDF → docs_paginas
    - Split em chunks
    - Tenta embeddings OpenAI → VectorStore (retriever vetorial)
    - Se não der, usa BM25 (lexical) como fallback

    Retorna:
      - retriever
      - n_pages
      - n_chunks
      - docs_paginas
    """
    # 1) Páginas
    docs_paginas: List[Document] = load_pdf_pages(pdf_path)
    n_pages = len(docs_paginas)

    # 2) Chunks (bom para livros)
    chunks: List[Document] = split_documents(
        docs_paginas,
        chunk_size=1200,
        chunk_overlap=200,
    )
    n_chunks = len(chunks)

    # 3) Embeddings OpenAI (se disponíveis)
    embeddings = get_embeddings()
    if embeddings is not None:
        retriever, n_chunks_vs = build_vectorstore(
            chunks,
            embeddings,
            k=6,
        )
        return retriever, n_pages, n_chunks_vs, docs_paginas

    # 4) Fallback BM25
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 6
    return bm25, n_pages, n_chunks, docs_paginas

