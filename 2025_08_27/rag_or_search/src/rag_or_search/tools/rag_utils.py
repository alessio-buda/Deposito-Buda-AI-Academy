from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import faiss
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings

# Chat model init (provider-agnostic, qui puntiamo a LM Studio via OpenAI-compatible)
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import getpass

# =========================
# Configurazione
# =========================

load_dotenv()

@dataclass
class Settings:
    # Persistenza FAISS
    persist_dir: str = "faiss_index_example"
    # Text splitting
    chunk_size: int = 1000
    chunk_overlap: int = 100
    # Retriever (MMR)
    search_type: str = "mmr"        # "mmr" o "similarity"
    k: int = 1                      # risultati finali
    fetch_k: int = 20               # candidati iniziali (per MMR)
    mmr_lambda: float = 1         # 0 = diversificazione massima, 1 = pertinenza massima
    # LM Studio (OpenAI-compatible)
    lmstudio_model_env: str = "MODEL"  # nome del modello in LM Studio, via env var



SETTINGS = Settings()


# =========================
# Componenti di base
# =========================

def get_embeddings():

    if not os.getenv("AZURE_API_KEY"):
        os.environ["AZURE_API_KEY"] = getpass.getpass(
            "Enter your AzureOpenAI API key: "
        )
    
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        openai_api_key=os.getenv("AZURE_API_KEY"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
    )


def get_llm_from_lmstudio(settings: Settings):
    """
    Inizializza un ChatModel puntando a LM Studio (OpenAI-compatible).
    Richiede:
      - OPENAI_BASE_URL (es. http://localhost:1234/v1)
      - OPENAI_API_KEY (placeholder qualsiasi, es. "not-needed")
      - LMSTUDIO_MODEL (nome del modello caricato in LM Studio)
    """
    base_url = os.getenv("AZURE_API_BASE")
    api_key = os.getenv("AZURE_API_KEY")
    api_version = os.getenv("AZURE_API_VERSION")
    model_name = os.getenv(settings.lmstudio_model_env)

    if not base_url or not api_key:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT e AZURE_OPENAI_KEY devono essere impostate per LM Studio."
        )
    if not model_name:
        raise RuntimeError(
            f"Imposta la variabile {settings.lmstudio_model_env} con il nome del modello caricato in LM Studio."
        )

    return init_chat_model(model_name, model_provider="azure_openai", api_key=api_key, api_version=api_version)

def load_documents(file_format, file_path):
    if file_format == "md":
        return load_md_documents(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
def load_md_documents(file_path: str) -> List[Document]:
    """
    Legge un file Markdown e restituisce una lista di Document (uno per ogni sezione separata da titoli).
    """
    documents = []
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    documents = [
        Document(
            page_content=section,
            metadata={"source": os.path.basename(file_path), "section": i}
        )
        for i, section in enumerate(content.split("---"), start=1)
        if section.strip()
    ]
    
    return documents
    

def simulate_corpus() -> List[Document]:
    """
    Crea un piccolo corpus di documenti in inglese con metadati e 'source' per citazioni.
    """
    docs = [
        Document(
            page_content=(
                "LangChain is a framework that helps developers build applications "
                "powered by Large Language Models (LLMs). It provides chains, agents, "
                "prompt templates, memory, and integrations with vector stores."
            ),
            metadata={"id": "doc1", "source": "intro-langchain.md"}
        ),
        Document(
            page_content=(
                "FAISS is a library for efficient similarity search and clustering of dense vectors. "
                "It supports exact and approximate nearest neighbor search and scales to millions of vectors."
            ),
            metadata={"id": "doc2", "source": "faiss-overview.md"}
        ),
        Document(
            page_content=(
                "Sentence-transformers like all-MiniLM-L6-v2 produce sentence embeddings suitable "
                "for semantic search, clustering, and information retrieval. The embedding size is 384."
            ),
            metadata={"id": "doc3", "source": "embeddings-minilm.md"}
        ),
        Document(
            page_content=(
                "A typical RAG pipeline includes indexing (load, split, embed, store) and "
                "retrieval+generation. Retrieval selects the most relevant chunks, and the LLM produces "
                "an answer grounded in those chunks."
            ),
            metadata={"id": "doc4", "source": "rag-pipeline.md"}
        ),
        Document(
            page_content=(
                "Maximal Marginal Relevance (MMR) balances relevance and diversity during retrieval. "
                "It helps avoid redundant chunks and improves coverage of different aspects."
            ),
            metadata={"id": "doc5", "source": "retrieval-mmr.md"}
        ),
    ]
    return docs


def split_documents(docs: List[Document], settings: Settings) -> List[Document]:
    """
    Applica uno splitting robusto ai documenti per ottimizzare il retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "\n\n", "\n", ". ", "? ", "! ", "; ", ": ",
            ", ", " ", ""  # fallback aggressivo
        ],
    )
    return splitter.split_documents(docs)


def build_faiss_vectorstore(chunks: List[Document], embeddings, persist_dir: str) -> FAISS:
    """
    Costruisce da zero un FAISS index (IndexFlatL2) e lo salva su disco.
    """
    # Determina la dimensione dell'embedding
    vs = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(persist_dir)
    return vs


def load_or_build_vectorstore(settings: Settings, embeddings, docs: List[Document]) -> FAISS:
    """
    Tenta il load di un indice FAISS persistente; se non esiste, lo costruisce e lo salva.
    """
    persist_path = Path(settings.persist_dir)
    index_file = persist_path / "index.faiss"
    meta_file = persist_path / "index.pkl"

    if index_file.exists() and meta_file.exists():
        # Dal 2024/2025 molte build richiedono il flag 'allow_dangerous_deserialization' per caricare pkl locali
        return FAISS.load_local(
            settings.persist_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )

    chunks = split_documents(docs, settings)
    return build_faiss_vectorstore(chunks, embeddings, settings.persist_dir)


def make_retriever(vector_store: FAISS, settings: Settings):
    """
    Configura il retriever. Con 'mmr' otteniamo risultati meno ridondanti e più coprenti.
    """
    if settings.search_type == "mmr":
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": settings.k, "fetch_k": settings.fetch_k, "lambda_mult": settings.mmr_lambda},
        )
    else:
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.k},
        )


def format_docs_for_prompt(docs: List[Document]) -> str:
    """
    Prepara il contesto per il prompt, includendo citazioni [source].
    """
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", f"doc{i}")
        lines.append(f"[source:{src}] {d.page_content}")
    return "\n\n".join(lines)


def rag_answer(question: str, chain) -> str:
    """
    Esegue la catena RAG per una singola domanda.
    """
    return chain.invoke(question)

def get_contexts_for_question(retriever, question: str, k: int) -> List[str]:
    """Ritorna i testi dei top-k documenti (chunk) usati come contesto."""
    docs = docs = retriever.invoke(question)[:k]
    return {d.metadata.get("source", f"doc{d.id}") : d.page_content for d in docs}

def rag_search(question: str, k: int):
    settings = SETTINGS
    
    settings.k = k  # aggiorna k dinamicamente

    # 1) Componenti
    embeddings = get_embeddings()
    llm = get_llm_from_lmstudio(settings)

    # 2) Dati simulati e indicizzazione (load or build)
    docs = simulate_corpus()
    vector_store = load_or_build_vectorstore(settings, embeddings, docs)

    # 3) Retriever ottimizzato
    retriever = make_retriever(vector_store, settings)
    
    retrieved_docs = get_contexts_for_question(retriever, question, k)
    
    return retrieved_docs
       