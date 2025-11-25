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
    "initial_greeting_done": False,
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

st.title("📄 PDF Assistant")

# ---------- Escolha de idioma ----------
lang_labels = {
    "pt": "Português",
    "it": "Italiano",
    "en": "English",
}

current_lang = st.session_state.get("app_lang", "pt")
lang_code = st.selectbox(
    "🌐 Idioma / Lingua / Language",
    ["pt", "it", "en"],
    index=["pt", "it", "en"].index(current_lang) if current_lang in ["pt", "it", "en"] else 0,
    format_func=lambda x: lang_labels.get(x, x),
)
st.session_state["app_lang"] = lang_code


def _hash_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


# ---------- Upload + Indexação ----------
if lang_code == "it":
    uploader_label = "Carica il tuo PDF"
elif lang_code == "en":
    uploader_label = "Upload your PDF"
else:
    uploader_label = "Envie o seu PDF"

pdf_file = st.file_uploader(uploader_label, type=["pdf"])

if pdf_file:
    raw = pdf_file.read()

    # usa o nome REAL do arquivo enviado
    uploaded_name = pdf_file.name
    temp_path = Path(uploaded_name)

    with open(temp_path, "wb") as f:
        f.write(raw)

    cache_key = _hash_bytes(raw)
    need_index = (st.session_state.get("vectordb_key") != cache_key)

    if need_index:
        # Reset ao trocar de PDF
        st.session_state["chat_history"] = []
        st.session_state["initial_greeting_done"] = False
        st.session_state["user_name"] = ""

        with st.spinner(
            "Indexing..." if lang_code == "en" else
            "Indicizzazione..." if lang_code == "it" else
            "Indexando..."
        ):
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
    if lang_code == "it":
        msg = "📂 Carica un PDF per iniziare."
    elif lang_code == "en":
        msg = "📂 Upload a PDF to start."
    else:
        msg = "📂 Envie um PDF para começar."
    st.info(msg)

