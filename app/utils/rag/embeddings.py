# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional
import os

from langchain_openai import OpenAIEmbeddings


def get_embeddings() -> Optional[OpenAIEmbeddings]:
    """
    Tenta criar embeddings da OpenAI.
    Se não houver OPENAI_API_KEY ou der erro, retorna None
    e o sistema usa fallback BM25.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        return OpenAIEmbeddings(model="text-embedding-3-large")
    except Exception:
        return None


