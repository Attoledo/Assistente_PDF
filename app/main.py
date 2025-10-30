# -*- coding: utf-8 -*-
import hashlib
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# ---------- Inicialização IMEDIATA do Session State ----------
DEFAULT_STATE = {
    "app_lang": "pt",
    "user_name": "",
    "chat_history": [],
    "just_named": False,
    "vectordb_key": None,
    "vectordb": None,
    "docs_paginas": [],
}
for key, val in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------- Imports que usam session_state (depois) ----------
from utils.pdf_index import indexar_pdf
from chat.assistente_dana import iniciar_assistente

# ---------- Configuração de página ----------
load_dotenv()
st.set_page_config(
    page_title="Assistente PDF / Assistant / Assistente",
    page_icon="📄",
    layout="wide"
)

# ✅ Título principal fixo + legenda
st.title("📄 PDF Assistant")


def _hash_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

# ---------- Upload + Indexação (com cache por hash) ----------
pdf_file = st.file_uploader("Upload / Envie / Carica il tuo PDF", type=["pdf"])

if pdf_file:
    raw = pdf_file.read()
    temp_path = Path("temp.pdf")
    with open(temp_path, "wb") as f:
        f.write(raw)

    cache_key = _hash_bytes(raw)
    need_index = (st.session_state.get("vectordb_key") != cache_key)

    if need_index:
        with st.spinner("Indexing / Indexando / Indicizzazione..."):
            retriever, n_pages, n_chunks, docs_pages = indexar_pdf(str(temp_path))
            st.session_state["vectordb"] = retriever
            st.session_state["docs_paginas"] = docs_pages
            st.session_state["vectordb_key"] = cache_key
            st.success(
                f"✅ Indexed!  "
                f"📄 Pages/Páginas/Pagine: {n_pages} • 🔹 Chunks: {n_chunks}"
            )

    iniciar_assistente(
        st,
        st.session_state.get("vectordb"),
        st.session_state.get("docs_paginas", []),
        nome_usuario=st.session_state.get("user_name", ""),
        k=5,
        incluir_vizinhas=1,
    )
else:
    st.info("📂 Upload a PDF to start • Envie um PDF para começar • Carica un PDF per iniziare.")