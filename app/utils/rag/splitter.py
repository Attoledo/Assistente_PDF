# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def make_text_splitter(
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> RecursiveCharacterTextSplitter:
    """
    Splitter robusto para livros/manuais grandes.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def split_documents(
    docs: List[Document],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Recebe a lista de páginas e devolve lista de chunks.
    """
    splitter = make_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    return chunks


