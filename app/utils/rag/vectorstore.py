# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore


def build_vectorstore(
    chunks: List[Document],
    embeddings: OpenAIEmbeddings,
    k: int = 6,
) -> Tuple[object, int]:
    """
    Cria um VectorStore em memória a partir dos chunks
    e devolve um retriever pronto + número de chunks.
    """
    vectordb = InMemoryVectorStore.from_documents(chunks, embedding=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    n_chunks = len(chunks)
    return retriever, n_chunks


