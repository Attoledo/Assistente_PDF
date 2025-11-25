# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Literal, Dict, Tuple
from pathlib import Path
import os

import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser


# ===========================
# Configuração do modelo (fixo)
# ===========================
LLM_PROVIDER_DEFAULT = "groq"                  # "groq" ou "openai"
LLM_MODEL_DEFAULT = "llama-3.1-8b-instant"    # modelo padrão Groq


def make_llm(
    provider: str = LLM_PROVIDER_DEFAULT,
    model: str = LLM_MODEL_DEFAULT,
    temperature: float = 0.2,
    max_tokens: int = 900,
    timeout: int = 90,
):
    """
    Cria o LLM fixo (Groq por padrão).
    """
    provider = (provider or "groq").lower()

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY não encontrada no ambiente.")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    # Default = Groq
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY não encontrada no ambiente.")
    return ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


# ===========================
# Idiomas (PT / IT / EN)
# ===========================
LANG: Dict[str, Dict[str, str]] = {
    "pt": {
        "ui_pdf_theme": "📚 Tema do PDF",
        "ui_theme_detected": "Tema detectado (edite se quiser)",
        "ui_caption": "O agente se comportará como especialista, com base no PDF enviado.",
        "ui_quick": "⚡ Tarefas rápidas",
        "ui_quick_page": "Resumo da página",
        "ui_quick_doc": "Resumo do documento",
        "ui_quick_gloss": "Glossário",
        "ui_quick_faq": "FAQ",
        "ui_quick_plan": "Plano de estudo",
        "ui_quick_exs": "Exercícios",
        "ui_go_page": "Ir para página (1-based)",
        "ui_read_page": "📄 Ler página",
        "ui_chat_ph": "Pergunte sobre o PDF (ex.: 'Explique o tópico X na página 10')",
        "ui_not_indexed": "Índice não carregado. Faça upload do PDF e aguarde a indexação.",
        "ui_no_pages": "Páginas do PDF não estão disponíveis. Refaça o upload para reindexar.",
        "ui_page_not_exist": "A página {page} não existe. Este PDF tem {total} páginas.",
        "ui_not_found": "Não encontrei trechos relevantes no documento para esta solicitação.",
        "ui_sources": "🔎 Trechos utilizados (fontes)",
        "your_name": "Seu nome",
        "clear_history": "🧹 Limpar histórico",
        "sys_tutor": (
            "Você é um especialista e tutor no tema do PDF indicado.\n"
            "PDF, Tema: {tema_pdf}\n\n"
            "Regras:\n"
            "1) Responda apenas com base no contexto; se faltar algo, diga explicitamente.\n"
            "2) Explique de forma clara e didática, com estrutura, passos e exemplos quando útil.\n"
            "3) Aprofunde, boas práticas, armadilhas comuns e próximos passos.\n"
            "4) Referencie páginas quando possível, por exemplo, 'ver págs. 62–63'.\n"
            "5) Seja conciso no essencial e útil nos detalhes.\n"
            "6) Trate sempre o usuário pelo nome `{nome}` quando fizer sentido.\n"
        ),
        "prompt_qa": (
            "Usuário: {nome}\n"
            "Pergunta: {pergunta}\n\n"
            "Contexto, trechos do PDF:\n{contexto}\n\n"
            "Responda como especialista no tema acima, seguindo as regras do sistema."
        ),
        "prompt_task": (
            "Usuário: {nome}\n"
            "Tarefa: {tarefa}\n"
            "Instruções: {instrucao}\n\n"
            "Contexto, trechos do PDF:\n{contexto}\n\n"
            "Produza a saída final, bem organizada e acionável, seguindo as regras do sistema."
        ),
        "task_page_instr": "Faça um resumo claro da página {page}, ideia central, pontos chave, termos e o que o leitor deve saber para aplicar.",
        "task_doc_instr": "Resumo executivo do PDF, objetivos, estrutura, conceitos centrais e aplicações. Inclua 'Como usar este PDF'.",
        "task_gloss_instr": "Glossário, termo, definição simples e onde aparece, página aproximada.",
        "task_faq_instr": "FAQ com 8–12 perguntas e respostas sobre dúvidas prováveis.",
        "task_plan_instr": "Plano de estudo, iniciante até avançado, com metas, prática e checkpoints.",
        "task_exs_instr": "Exercícios práticos, 5–10, com objetivo, passos e critério de sucesso.",
        "word_page_patterns": [r"\bp[aá]g(?:ina)?\.?\s*(\d+)\b", r"\bp[aá]gina\.?\s*(\d+)\b"],
    },
    "it": {
        "ui_pdf_theme": "📚 Tema del PDF",
        "ui_theme_detected": "Tema rilevato (modificabile)",
        "ui_caption": "L'agente si comporta come esperto, basandosi sul PDF caricato.",
        "ui_quick": "⚡ Task rapidi",
        "ui_quick_page": "Riassunto pagina",
        "ui_quick_doc": "Riassunto documento",
        "ui_quick_gloss": "Glossario",
        "ui_quick_faq": "FAQ",
        "ui_quick_plan": "Piano di studio",
        "ui_quick_exs": "Esercizi",
        "ui_go_page": "Vai a pagina (1-based)",
        "ui_read_page": "📄 Leggi pagina",
        "ui_chat_ph": "Fai una domanda sul PDF (es.: 'Spiega il punto X a pagina 10')",
        "ui_not_indexed": "Indice non caricato. Carica un PDF e attendi l'indicizzazione.",
        "ui_no_pages": "Le pagine del PDF non sono disponibili. Ricarica per reindicizzare.",
        "ui_page_not_exist": "La pagina {page} non esiste. Questo PDF ha {total} pagine.",
        "ui_not_found": "Non ho trovato estratti rilevanti nel documento per questa richiesta.",
        "ui_sources": "🔎 Estratti usati (fonti)",
        "your_name": "Il tuo nome",
        "clear_history": "🧹 Pulisci storico",
        "sys_tutor": (
            "Sei un esperto e tutor nel tema del PDF indicato.\n"
            "PDF, Tema: {tema_pdf}\n\n"
            "Regole:\n"
            "1) Rispondi solo in base al contesto; se manca qualcosa, dillo chiaramente.\n"
            "2) Spiega in modo chiaro e didattico, con struttura, passi ed esempi quando utile.\n"
            "3) Approfondisci, buone pratiche, errori comuni e prossimi passi.\n"
            "4) Cita sempre le pagine quando possibile (es.: 'vedi pagg. 62–63').\n"
            "5) Sii conciso e utile.\n"
            "6) Rivolgiti sempre all’utente per nome `{nome}` quando è naturale.\n"
        ),
        "prompt_qa": (
            "Utente: {nome}\n"
            "Domanda: {pergunta}\n\n"
            "Contesto, estratti dal PDF:\n{contexto}\n\n"
            "Rispondi come esperto del tema sopra, seguendo le regole di sistema."
        ),
        "prompt_task": (
            "Utente: {nome}\n"
            "Task: {tarefa}\n"
            "Istruzioni: {instrucao}\n\n"
            "Contesto, estratti dal PDF:\n{contexto}\n\n"
            "Produci l'output finale, ben organizzato e azionabile, seguendo le regole di sistema."
        ),
        "task_page_instr": "Riassumi chiaramente la pagina {page}, idea centrale, punti chiave, termini e ciò che serve per applicare.",
        "task_doc_instr": "Sommario esecutivo, obiettivi, struttura, concetti chiave e applicazioni. Aggiungi 'Come usare questo PDF'.",
        "task_gloss_instr": "Glossario, termine, definizione semplice e dove appare, pagina indicativa.",
        "task_faq_instr": "FAQ con 8–12 Q&A chiare su dubbi probabili.",
        "task_plan_instr": "Piano di studio, base fino avanzato, con obiettivi, pratica e checkpoint.",
        "task_exs_instr": "Esercizi pratici, 5–10, con obiettivo, passi e criterio di successo.",
        "word_page_patterns": [r"\bpagina\.?\s*(\d+)\b", r"\bpag\.?\s*(\d+)\b"],
    },
    "en": {
        "ui_pdf_theme": "📚 PDF Theme",
        "ui_theme_detected": "Detected theme (you may edit)",
        "ui_caption": "The agent behaves as a specialist, based on the uploaded PDF.",
        "ui_quick": "⚡ Quick tasks",
        "ui_quick_page": "Page summary",
        "ui_quick_doc": "Document summary",
        "ui_quick_gloss": "Glossary",
        "ui_quick_faq": "FAQ",
        "ui_quick_plan": "Study plan",
        "ui_quick_exs": "Exercises",
        "ui_go_page": "Go to page (1-based)",
        "ui_read_page": "📄 Read page",
        "ui_chat_ph": "Ask about the PDF (e.g., 'Explain topic X on page 10')",
        "ui_not_indexed": "Index not loaded. Upload a PDF and wait for indexing.",
        "ui_no_pages": "PDF pages are unavailable. Re-upload to re-index.",
        "ui_page_not_exist": "Page {page} does not exist. This PDF has {total} pages.",
        "ui_not_found": "No relevant excerpts found in the document for this request.",
        "ui_sources": "🔎 Used excerpts (sources)",
        "your_name": "Your name",
        "clear_history": "🧹 Clear history",
        "sys_tutor": (
            "You are a specialist and tutor in the theme of the provided PDF.\n"
            "PDF, Theme: {tema_pdf}\n\n"
            "Rules:\n"
            "1) Answer only based on the provided context; if something is missing, say it clearly.\n"
            "2) Explain clearly and pedagogically, with structure, steps and examples when useful.\n"
            "3) Go deeper, best practices, common pitfalls and next steps.\n"
            "4) Always reference pages when possible (e.g., 'see pp. 62–63').\n"
            "5) Be concise and useful.\n"
            "6) Always address the user by their name `{nome}` when appropriate.\n"
        ),
        "prompt_qa": (
            "User: {nome}\n"
            "Question: {pergunta}\n\n"
            "Context, PDF excerpts:\n{contexto}\n\n"
            "Answer as a specialist in the theme above, following the system rules."
        ),
        "prompt_task": (
            "User: {nome}\n"
            "Task: {tarefa}\n"
            "Instructions: {instrucao}\n\n"
            "Context, PDF excerpts:\n{contexto}\n\n"
            "Produce the final output, well organized and actionable, following the system rules."
        ),
        "task_page_instr": "Summarize page {page}, main idea, key points, terms, and what the reader must know to apply.",
        "task_doc_instr": "Executive summary, goals, structure, core concepts and applications. Add 'How to use this PDF'.",
        "task_gloss_instr": "Glossary, term, simple definition and where it appears, approximate page.",
        "task_faq_instr": "FAQ with 8–12 clear Q&A for likely doubts.",
        "task_plan_instr": "Study plan, beginner to advanced, with goals, practice and checkpoints.",
        "task_exs_instr": "Hands-on exercises, 5–10, with objective, steps and success criteria.",
        "word_page_patterns": [r"\bpage\.?\s*(\d+)\b", r"\bpg\.?\s*(\d+)\b"],
    },
}

QuickTask = Literal["resumo_pagina", "resumo_documento", "glossario", "faq", "plano_estudo", "exercicios"]


# ===========================
# Session State seguro
# ===========================
DEFAULT_STATE_LOCAL = {
    "app_lang": "pt",
    "user_name": "",
    "chat_history": [],
    "just_named": False,
    "initial_greeting_done": False,
}

def SAFE_GET(key: str, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state.get(key, default)

def SAFE_SET(key: str, value):
    st.session_state[key] = value
    return value

def ensure_session_defaults():
    for k, v in DEFAULT_STATE_LOCAL.items():
        SAFE_GET(k, v)


# ===========================
# Utils de contexto
# ===========================
def _compactar_docs(docs: List[Document], limite_chars: int = 5200) -> str:
    parts, total = [], 0
    for d in docs:
        t = (d.page_content or "").strip()
        if not t:
            continue
        if total + len(t) > limite_chars:
            break
        parts.append(t)
        total += len(t)
    return "\n\n---\n\n".join(parts)

def _normalize(s: str) -> str:
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii")


def _detectar_tema_pdf(docs_paginas: List[Document]) -> str:
    """
    Detecção de tema mais robusta:
    - analisa as primeiras páginas
    - ignora linhas de endereço, empresa, lixo (iiiiii, etc.)
    - tenta encontrar algo que pareça título de manual/livro
    """
    if not docs_paginas:
        return "tema do PDF / tema del PDF / theme of the PDF"

    candidates: List[str] = []

    max_pages = min(5, len(docs_paginas))
    for i in range(max_pages):
        text = (docs_paginas[i].page_content or "").strip()
        if not text:
            continue
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in lines[:40]:  # primeiras linhas de cada página
            ln_norm = _normalize(ln).lower()

            # filtros de ruído
            if len(ln) < 5 or len(ln) > 120:
                continue
            # linha quase só pontuação ou dígito
            if sum(ch.isalnum() for ch in ln) < len(ln) * 0.4:
                continue
            # muito repetitiva, tipo "iiiiiiiii"
            if len(set(ln.replace(" ", ""))) == 1:
                continue
            # coisas típicas de endereço/empresa
            lixo = [
                "via ", "s.r.l", "srl", "s.p.a", "spa",
                "lumezzane", "automazioni industriali",
                "@", ".it", ".com", "tel.", "fax",
            ]
            if any(w in ln_norm for w in lixo):
                continue

            # palavras típicas de título de manual
            palavras_titulo = [
                "manuale", "manuale d'uso", "manuale applicativo",
                "robot", "motoman", "applicativo", "uso", "introduzione",
                "user manual", "operation manual", "guia", "manual",
            ]
            score = 0
            for w in palavras_titulo:
                if w in ln_norm:
                    score += 1

            # se tem palavras de título, é forte candidato
            if score > 0:
                candidates.append(ln)
            else:
                # se parecer uma frase decente, também serve
                if ln[0].isupper() and " " in ln:
                    candidates.append(ln)

    if not candidates:
        return "tema do PDF / tema del PDF / theme of the PDF"

    # remove duplicados mantendo ordem
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    # junta no máximo 2 linhas relevantes
    tema = " — ".join(uniq[:2])
    return tema or "tema do PDF / tema del PDF / theme of the PDF"


def _extrair_pagina(pergunta: str, patt_list: List[str]) -> Optional[int]:
    if not pergunta:
        return None
    norm = _normalize(pergunta.lower())
    for pat in patt_list:
        m = re.search(_normalize(pat), norm, flags=re.IGNORECASE)
        if m:
            try:
                return max(int(m.group(1)) - 1, 0)  # 1-based -> 0-based
            except ValueError:
                continue
    return None

def _contexto_por_pagina(docs_paginas: List[Document], pagina0: int, vizinhas: int = 0) -> List[Document]:
    n = len(docs_paginas)
    if n == 0:
        return []
    start = max(pagina0 - vizinhas, 0)
    end = min(pagina0 + vizinhas, n - 1)
    return [docs_paginas[i] for i in range(start, end + 1)]

def _tem_texto(docs: List[Document]) -> bool:
    return any((d.page_content or "").strip() for d in docs)

def _instrucao_tarefa(task: QuickTask, pagina: Optional[int], L: Dict[str, str]) -> str:
    page_h = (pagina + 1) if pagina is not None else "?"
    if task == "resumo_pagina":
        return L["task_page_instr"].format(page=page_h)
    if task == "resumo_documento":
        return L["task_doc_instr"]
    if task == "glossario":
        return L["task_gloss_instr"]
    if task == "faq":
        return L["task_faq_instr"]
    if task == "plano_estudo":
        return L["task_plan_instr"]
    if task == "exercicios":
        return L["task_exs_instr"]
    return L["task_doc_instr"]

def _montar_contexto(
    pergunta: Optional[str],
    docs_paginas: List[Document],
    retriever_or_vector,
    k: int,
    incluir_vizinhas: int,
    pagina_forcada: Optional[int],
    patt_list: List[str],
) -> Tuple[List[Document], List[tuple], Optional[int]]:
    pagina0 = pagina_forcada
    if pagina0 is None and pergunta:
        pagina0 = _extrair_pagina(pergunta, patt_list)

    if pagina0 is not None:
        if 0 <= pagina0 < len(docs_paginas):
            docs = _contexto_por_pagina(docs_paginas, pagina0, vizinhas=incluir_vizinhas)
            info = [
                (d.metadata.get("page", "?"), d.metadata.get("source", d.metadata.get("file_path", "")))
                for d in docs
            ]
            return docs, info, pagina0
        else:
            return [], [], pagina0

    if hasattr(retriever_or_vector, "invoke"):
        docs = retriever_or_vector.invoke(pergunta or "")
    else:
        retriever = retriever_or_vector.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(pergunta or "")

    info = [
        (d.metadata.get("page", "?"), d.metadata.get("source", d.metadata.get("file_path", "")))
        for d in docs
    ]
    return docs, info, None


def _extrair_primeiro_nome(resposta: str) -> str:
    """
    Extrai apenas o primeiro nome do texto, ignorando o resto.
    Exemplo: 'Olá, me chamo Jonas, tudo bem?' -> 'Jonas'
    """
    if not resposta:
        return ""

    original = resposta.strip()
    lower = original.lower()

    patterns = [
        "me chamo",
        "meu nome é",
        "meu nome e",
        "eu sou",
        "mi chiamo",
        "sono",
        "my name is",
        "i am",
    ]

    recorte = original
    for p in patterns:
        if p in lower:
            idx = lower.index(p) + len(p)
            recorte = original[idx:].strip()
            break

    for ch in [",", ".", "!", "?", ";", ":"]:
        recorte = recorte.replace(ch, " ")

    partes = recorte.split()
    if not partes:
        return ""

    stopwords_iniciais = {
        "ola", "olá", "oi", "ciao", "hello", "hi", "hey",
        "buongiorno", "buonasera", "bom", "boa",
    }

    for palavra in partes:
        if palavra.lower() in stopwords_iniciais:
            continue
        return palavra.strip()

    return partes[0].strip()


# ===========================
# Intenções especiais (router)
# ===========================
def detectar_intencao(pergunta: str) -> str:
    """
    Detecta intenções estruturais, como:
    - contar_paginas
    - nome_pdf
    Retorna 'normal' se não detectar nada especial.
    """
    if not pergunta:
        return "normal"

    p = _normalize(pergunta.lower())

    # Contar páginas
    pads_pag = [
        "quantas paginas", "quantas páginas", "numero de paginas", "numero de páginas",
        "número de paginas", "número de páginas", "total de paginas", "total de páginas",
        "quante pagine", "numero di pagine", "numero pagine", "totale pagine",
        "how many pages", "number of pages", "total pages", "page count",
    ]
    if any(k in p for k in pads_pag):
        return "contar_paginas"

    # Nome do PDF / arquivo
    pads_nome = [
        "nome do arquivo", "nome do ficheiro", "nome do pdf",
        "come si chiama il file", "nome del file", "nome del pdf",
        "file name", "name of the file", "pdf name", "name of the pdf",
    ]
    if any(k in p for k in pads_nome):
        return "nome_pdf"

    return "normal"


QuickTaskType = Literal["resumo_pagina", "resumo_documento", "glossario", "faq", "plano_estudo", "exercicios"]


def iniciar_assistente(
    st,
    vectordb,
    docs_paginas: List[Document],
    nome_usuario: str = "",
    k: int = 5,
    incluir_vizinhas: int = 1,
):
    ensure_session_defaults()

    # Parâmetros fixos ideais
    k = 6
    incluir_vizinhas = 1

    if nome_usuario and not SAFE_GET("user_name"):
        SAFE_SET("user_name", nome_usuario)

    lang_code = SAFE_GET("app_lang", "pt")
    L = LANG.get(lang_code, LANG["pt"])

    if vectordb is None:
        st.error(L["ui_not_indexed"])
        return
    if not docs_paginas:
        st.error(L["ui_no_pages"])
        return

    # Tema + nome de arquivo
    tema_pdf_auto = _detectar_tema_pdf(docs_paginas)

    file_name = "PDF"
    try:
        if docs_paginas and isinstance(docs_paginas[0].metadata, dict):
            file_name = (
                docs_paginas[0].metadata.get("source")
                or Path(docs_paginas[0].metadata.get("file_path", "")).name
                or "PDF"
            )
    except Exception:
        pass

    tema_final = tema_pdf_auto.strip() or tema_pdf_auto
    if lang_code == "it":
        subtitulo = f"### 📘 Esperto in **{tema_final}**, basato su *{file_name}*"
    elif lang_code == "en":
        subtitulo = f"### 📘 Expert in **{tema_final}** based on *{file_name}*"
    else:
        subtitulo = f"### 📘 Especialista em **{tema_final}** baseado em *{file_name}*"

    st.markdown(subtitulo)

    # Sidebar amigável
    with st.sidebar:
        st.subheader(L["ui_pdf_theme"])
        tema_pdf = st.text_input(L["ui_theme_detected"], value=tema_pdf_auto)
        st.caption(L["ui_caption"])

        if st.button(L["clear_history"]):
            SAFE_SET("chat_history", [])
            SAFE_SET("initial_greeting_done", False)
            SAFE_SET("user_name", "")
            st.rerun()

        st.subheader(L["ui_quick"])
        col_a, col_b = st.columns(2)
        with col_a:
            bt_res_pag = st.button(L["ui_quick_page"])
            bt_gloss   = st.button(L["ui_quick_gloss"])
            bt_exs     = st.button(L["ui_quick_exs"])
        with col_b:
            bt_res_doc = st.button(L["ui_quick_doc"])
            bt_faq     = st.button(L["ui_quick_faq"])
            bt_plano   = st.button(L["ui_quick_plan"])

    tema_final = (tema_pdf or tema_pdf_auto).strip() or tema_pdf_auto

    # Saudação inicial da Dana
    if not SAFE_GET("initial_greeting_done", False):
        if lang_code == "it":
            first_msg = (
                f"Ciao, io sono **Dana** 👋\n\n"
                f"Sono qui per aiutarti con tutte le tue domande su **{file_name}**.\n\n"
                "Per cominciare, come posso chiamarti?"
            )
        elif lang_code == "en":
            first_msg = (
                f"Hi, I'm **Dana** 👋\n\n"
                f"I'm here to help you with any questions about **{file_name}**.\n\n"
                "To get started, how should I call you?"
            )
        else:
            first_msg = (
                f"Olá, eu sou a **Dana** 👋\n\n"
                f"Estou aqui para tirar todas as suas dúvidas sobre **{file_name}**.\n\n"
                "Para começar, como posso te chamar?"
            )

        with st.chat_message("assistant"):
            st.markdown(first_msg)
        hist = SAFE_GET("chat_history", [])
        hist.append({"role": "assistant", "content": first_msg})
        SAFE_SET("chat_history", hist)
        SAFE_SET("initial_greeting_done", True)

    # Navegação por página
    col1, col2 = st.columns([1, 3])
    with col1:
        page_num = st.number_input(L["ui_go_page"], min_value=1, max_value=len(docs_paginas), value=1, step=1)
    with col2:
        go_btn = st.button(L["ui_read_page"])

    # Histórico
    for msg in SAFE_GET("chat_history", []):
        with st.chat_message(msg.get("role", "assistant")):
            st.markdown(msg.get("content", ""))

    # Entrada do usuário
    pergunta = st.chat_input(L["ui_chat_ph"])

    # Primeira mensagem, captura nome
    if pergunta and not SAFE_GET("user_name", ""):
        nome_extraido = _extrair_primeiro_nome(pergunta)
        if nome_extraido:
            SAFE_SET("user_name", nome_extraido)

        hist = SAFE_GET("chat_history", [])
        hist.append({"role": "user", "content": pergunta})
        SAFE_SET("chat_history", hist)

        if nome_extraido:
            if lang_code == "it":
                reply = (
                    f"{nome_extraido}, perfetto! Ora puoi chiedermi qualsiasi dubbio che hai su **{file_name}**. "
                    "Scrivi pure la tua prima domanda quando vuoi. 🙂"
                )
            elif lang_code == "en":
                reply = (
                    f"{nome_extraido}, great! Now you can ask me any question you have about **{file_name}**. "
                    "Type your first question whenever you're ready. 🙂"
                )
            else:
                reply = (
                    f"{nome_extraido}, perfeito! Agora você pode me perguntar qualquer coisa sobre **{file_name}**. "
                    "Escreve sua primeira pergunta quando quiser. 🙂"
                )
        else:
            if lang_code == "it":
                reply = "Perfetto! Ora puoi scrivermi la tua prima domanda sul PDF. 🙂"
            elif lang_code == "en":
                reply = "Great! Now you can type your first question about the PDF. 🙂"
            else:
                reply = "Perfeito! Agora você pode escrever sua primeira pergunta sobre o PDF. 🙂"

        with st.chat_message("assistant"):
            st.markdown(reply)
        hist = SAFE_GET("chat_history", [])
        hist.append({"role": "assistant", "content": reply})
        SAFE_SET("chat_history", hist)
        return

    # Quick tasks / navegação
    quick_task: Optional[QuickTaskType] = None
    pagina_forcada: Optional[int] = None

    if go_btn:
        pagina_forcada = page_num - 1
        pergunta = f"pagina {page_num}" if lang_code != "en" else f"page {page_num}"

    if st.session_state.get("_last_quick_task_clicked") is None:
        st.session_state["_last_quick_task_clicked"] = ""

    if bt_res_pag:
        quick_task = "resumo_pagina"
        pagina_forcada = page_num - 1
        if not pergunta:
            pergunta = f"pagina {page_num}" if lang_code != "en" else f"page {page_num}"
    elif bt_res_doc:
        quick_task = "resumo_documento"
        if not pergunta:
            pergunta = (
                "resumo do documento" if lang_code == "pt"
                else "riassunto del documento" if lang_code == "it"
                else "document summary"
            )
    elif bt_gloss:
        quick_task = "glossario"
        if not pergunta:
            pergunta = (
                "glossário" if lang_code == "pt"
                else "glossario" if lang_code == "it"
                else "glossary"
            )
    elif bt_faq:
        quick_task = "faq"
        if not pergunta:
            pergunta = "faq"
    elif bt_plano:
        quick_task = "plano_estudo"
        if not pergunta:
            pergunta = (
                "plano de estudo" if lang_code == "pt"
                else "piano di studio" if lang_code == "it"
                else "study plan"
            )
    elif bt_exs:
        quick_task = "exercicios"
        if not pergunta:
            pergunta = (
                "exercícios" if lang_code == "pt"
                else "esercizi" if lang_code == "it"
                else "exercises"
            )

    if not pergunta and not quick_task:
        return

    # Intenção especial (páginas, nome do PDF, etc.)
    intencao = detectar_intencao(pergunta or "")

    if intencao == "contar_paginas":
        total = len(docs_paginas)
        if lang_code == "it":
            resposta = f"Questo PDF ha **{total} pagine**."
        elif lang_code == "en":
            resposta = f"This PDF has **{total} pages**."
        else:
            resposta = f"Este PDF possui **{total} páginas**."

        with st.chat_message("assistant"):
            st.markdown(resposta)
        hist = SAFE_GET("chat_history", [])
        hist.append({"role": "assistant", "content": resposta})
        SAFE_SET("chat_history", hist)
        return

    if intencao == "nome_pdf":
        if lang_code == "it":
            resposta = f"Il nome del file è **{file_name}**."
        elif lang_code == "en":
            resposta = f"The file name is **{file_name}**."
        else:
            resposta = f"O nome do arquivo é **{file_name}**."

        with st.chat_message("assistant"):
            st.markdown(resposta)
        hist = SAFE_GET("chat_history", [])
        hist.append({"role": "assistant", "content": resposta})
        SAFE_SET("chat_history", hist)
        return

    # Historiza pergunta (agora como pergunta normal no histórico)
    hist = SAFE_GET("chat_history", [])
    uname = SAFE_GET("user_name", "")
    hist.append({"role": "user", "content": f"({uname}) {pergunta}"})
    SAFE_SET("chat_history", hist)

    # Monta contexto
    patt_list = L["word_page_patterns"]
    docs, info_sources, pagina0 = _montar_contexto(
        pergunta, docs_paginas, vectordb, k, incluir_vizinhas, pagina_forcada, patt_list
    )

    if pagina0 is not None and (pagina0 < 0 or pagina0 >= len(docs_paginas)):
        msg = L["ui_page_not_exist"].format(page=pagina0 + 1, total=len(docs_paginas))
        with st.chat_message("assistant"):
            st.markdown(msg)
        hist = SAFE_GET("chat_history", [])
        hist.append({"role": "assistant", "content": msg})
        SAFE_SET("chat_history", hist)
        return

    if not docs or not _tem_texto(docs):
        msg = L["ui_not_found"]
        with st.chat_message("assistant"):
            st.markdown(msg)
        hist = SAFE_GET("chat_history", [])
        hist.append({"role": "assistant", "content": msg})
        SAFE_SET("chat_history", hist)
        return

    contexto = _compactar_docs(docs, limite_chars=5200)

    SYSTEM_TUTOR = L["sys_tutor"]
    PROMPT_QA = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_TUTOR), MessagesPlaceholder("history"), ("user", L["prompt_qa"])]
    )
    PROMPT_TASK = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_TUTOR), MessagesPlaceholder("history"), ("user", L["prompt_task"])]
    )
    history_msgs = [
        {"role": m.get("role", "assistant"), "content": m.get("content", "")}
        for m in SAFE_GET("chat_history", [])
    ]

    try:
        llm = make_llm()
    except Exception as e:
        st.error(f"Erro ao inicializar o modelo ({LLM_PROVIDER_DEFAULT}/{LLM_MODEL_DEFAULT}): {e}")
        return

    if quick_task:
        instr = _instrucao_tarefa(quick_task, pagina0, L)
        chain = (
            RunnableMap({
                "tema_pdf": lambda _: tema_final,
                "nome":     lambda _: SAFE_GET("user_name", ""),
                "tarefa":   lambda _: quick_task,
                "instrucao":lambda _: instr,
                "contexto": lambda _: contexto,
                "pergunta": lambda _: "",
                "history":  lambda _: history_msgs,
            })
            | PROMPT_TASK
            | llm
            | StrOutputParser()
        )
    else:
        chain = (
            RunnableMap({
                "tema_pdf": lambda _: tema_final,
                "nome":     lambda _: SAFE_GET("user_name", ""),
                "pergunta": lambda _: pergunta,
                "contexto": lambda _: contexto,
                "tarefa":   lambda _: "",
                "instrucao":lambda _: "",
                "history":  lambda _: history_msgs,
            })
            | PROMPT_QA
            | llm
            | StrOutputParser()
        )

    try:
        resposta = chain.invoke({})
    except Exception as e:
        resposta = f"Model call failed: {e}"

    with st.chat_message("assistant"):
        st.markdown(resposta)
    hist = SAFE_GET("chat_history", [])
    hist.append({"role": "assistant", "content": resposta})
    SAFE_SET("chat_history", hist)

    # Fontes usadas
    with st.expander(L["ui_sources"]):
        for pag, src in info_sources:
            try:
                human_p = int(pag) + 1
            except Exception:
                human_p = pag
            st.markdown(f"- **Page/Página/Pagina**: {human_p} — **Source/Fonte/Fonte**: {src}")


