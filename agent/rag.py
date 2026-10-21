"""Support-doc retrieval with LangChain + Chroma.

Each markdown doc has a small front-matter block with metadata. Dates are stored as integers
(YYYYMMDD) because Chroma's $gte/$lte filters only work on numbers. A doc with no valid_to
gets 99991231, meaning "still valid".
"""
from datetime import date
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent import config

OPEN_ENDED = 99991231
COLLECTION = "support_docs"


def _date_int(value: str) -> int:
    return int(value.replace("-", "")) if value else OPEN_ENDED


def parse_doc(path: Path) -> Document:
    """Split '---' front matter from the body and turn it into Chroma-friendly metadata."""
    _, header, body = path.read_text().split("---", 2)
    meta = {}
    for line in header.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return Document(
        page_content=body.strip(),
        metadata={
            "source": path.name,
            "doc_type": meta["doc_type"],
            "model": meta["model"],
            "year": meta["year"],
            "valid_from": _date_int(meta["valid_from"]),
            "valid_to": _date_int(meta["valid_to"]),
        },
    )


def build_filter(model: str | None, year: int | None, today: date) -> dict:
    """Only docs valid today, and only for the given model/year (or docs that apply to all)."""
    today_int = int(today.strftime("%Y%m%d"))
    conditions = [
        {"valid_from": {"$lte": today_int}},
        {"valid_to": {"$gte": today_int}},
    ]
    if model:
        conditions.append({"model": {"$in": [model, "all"]}})
    if year:
        conditions.append({"year": {"$in": [str(year), "all"]}})
    return {"$and": conditions}


def _default_embeddings() -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL)


def build_index(embeddings: Embeddings | None = None, persist_dir: Path = config.CHROMA_DIR) -> Chroma:
    """(Re)build the vector store from data/docs. Run once: python -m agent.rag"""
    docs = [parse_doc(p) for p in sorted(config.DOCS_DIR.glob("*.md"))]
    chunks = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60).split_documents(docs)
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings or _default_embeddings(),
        persist_directory=str(persist_dir),
    )
    store.reset_collection()
    store.add_documents(chunks)
    return store


_store: Chroma | None = None


def get_store() -> Chroma:
    global _store
    if _store is None:
        _store = Chroma(
            collection_name=COLLECTION,
            embedding_function=_default_embeddings(),
            persist_directory=str(config.CHROMA_DIR),
        )
    return _store


def set_store(store: Chroma) -> None:
    """Used by tests to swap in a store with fake embeddings."""
    global _store
    _store = store


def search(query: str, model: str | None = None, year: int | None = None,
           today: date | None = None, k: int = 3) -> list[dict]:
    flt = build_filter(model, year, today or date.today())
    results = get_store().similarity_search(query, k=k, filter=flt)
    return [{"source": d.metadata["source"], "content": d.page_content} for d in results]


if __name__ == "__main__":
    store = build_index()
    print(f"Indexed {store._collection.count()} chunks into {config.CHROMA_DIR}")
