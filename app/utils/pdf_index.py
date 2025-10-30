# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple, List
from pathlib import Path
import os

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # <-- novo import

# Vetorial (OpenAI) ou Fallback BM25
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.retrievers import BM25Retriever


def _carregar_paginas(pdf_path: str) -> List[Document]:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()  # um Document por página
    # garantir metadados page 0-based
    for i, d in enumerate(pages):
        d.metadata = dict(d.metadata or {})
        d.metadata["page"] = i  # 0-based
        d.metadata.setdefault("source", Path(pdf_path).name)
        d.metadata["file_path"] = str(Path(pdf_path).resolve())
    return pages


def _splitar(docs: List[Document], chunk_size=1000, overlap=150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_documents(docs)
    return chunks


def _tentar_embeddings() -> OpenAIEmbeddings | None:
    # Usa OpenAI Embeddings se houver chave
    if os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAIEmbeddings(model="text-embedding-3-large")
        except Exception:
            return None
    return None


def indexar_pdf(pdf_path: str) -> Tuple[object, int, int, List[Document]]:
    """
    Retorna:
      - retriever (ou vectorstore com .as_retriever): objeto compatível,
      - n_pages,
      - n_chunks,
      - docs_paginas (lista Document por página)
    """
    docs_paginas = _carregar_paginas(pdf_path)
    chunks = _splitar(docs_paginas, 1000, 150)

    embeddings = _tentar_embeddings()
    if embeddings:
        # InMemoryVectorStore não precisa de faiss/chroma e funciona em memória
        vectordb = InMemoryVectorStore.from_documents(chunks, embedding=embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": 6})
        return retriever, len(docs_paginas), len(chunks), docs_paginas

    # Fallback sem embeddings: BM25 (lexical)
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = 6
    return retriever, len(docs_paginas), len(chunks), docs_paginas