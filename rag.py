import json
import logging
import re
import time
from collections import OrderedDict

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from openai import AzureOpenAI

import config

log = logging.getLogger("rag")

SYSTEM_PROMPT = """You are an enterprise knowledge assistant. Answer ONLY from the provided context.
Rules:
- If the context is insufficient, reply with exactly: "I don't have enough information in the knowledge base to answer that." — and cite nothing.
- If the question is ambiguous and the context contains several different things it could refer to, do not pick one silently. Ask a short clarifying question listing the possible meanings and cite nothing.
- Otherwise cite sources inline as [Title] immediately after the claim they support.
- Be concise and factual."""

REWRITE_PROMPT = """Rewrite the user's follow-up question as a standalone search query.
Use the conversation to resolve pronouns and references like "what about" or "the limit".
Return only the rewritten query, no quotes, no explanation."""

_credential = DefaultAzureCredential()
_search = SearchClient(config.SEARCH_ENDPOINT, config.SEARCH_INDEX, credential=_credential)
_chat = AzureOpenAI(
    azure_endpoint=config.OPENAI_ENDPOINT,
    azure_ad_token_provider=get_bearer_token_provider(
        _credential, "https://cognitiveservices.azure.com/.default"
    ),
    api_version="2024-10-21",
)

def rewrite_query(question: str, history: list[dict]) -> str:
    if not history or config.BASELINE:
        return question
    resp = _chat.chat.completions.create(
        model=config.OPENAI_CHAT_DEPLOYMENT,
        messages=[{"role": "system", "content": REWRITE_PROMPT}]
        + history
        + [{"role": "user", "content": question}],
    )
    rewritten = (resp.choices[0].message.content or "").strip()
    return rewritten or question

_parent_map: dict[str, str] | None = None

def _parent_ids() -> dict[str, str]:
    """Cached title -> parent_id lookup from the index."""
    global _parent_map
    if _parent_map is None:
        _parent_map = {}
        for r in _search.search(search_text="*", top=1000, select=["title", "parent_id"]):
            _parent_map.setdefault(r["title"], r["parent_id"])
    return _parent_map

def _role_filter(role: str | None) -> str | None:
    role = (role or "all").strip().lower()
    if config.BASELINE:
        return None
    if role in ("", "all", "admin"):
        return None
    titles = config.ROLE_TITLES.get(role)
    if not titles:
        return None
    parents = _parent_ids()
    ids = [parents[t] for t in sorted(titles) if t in parents]
    if not ids:

        return "chunk_id eq '__deny_all__'"
    return " or ".join(f"parent_id eq '{p}'" for p in ids)

_version_re = re.compile(r"^(.*?)(19\d{2}|20\d{2})(\.[A-Za-z0-9]+)$")
_year_in_text_re = re.compile(r"\b(19\d{2}|20\d{2})\b")

def _split_version(title: str) -> tuple[str, int]:
    m = _version_re.match(title)
    if not m:
        return title, 0
    return m.group(1) + m.group(3), int(m.group(2))

def prefer_newest(question: str, chunks: list[dict]) -> list[dict]:
    """Keep only the newest version of each document, unless the question names a year."""
    if _year_in_text_re.search(question):
        return chunks
    groups: dict[str, list[tuple[int, dict]]] = {}
    for c in chunks:
        base, year = _split_version(c["title"])
        groups.setdefault(base, []).append((year, c))
    out: list[dict] = []
    for items in groups.values():
        newest = max(year for year, _ in items)
        out.extend(c for year, c in items if year == newest)
    return out

def retrieve(search_query: str, role: str | None = None) -> list[dict]:
    results = _search.search(
        search_text=search_query,
        vector_queries=[
            VectorizableTextQuery(
                text=search_query, k_nearest_neighbors=config.VECTOR_K, fields="text_vector"
            )
        ],
        query_type="semantic",
        semantic_configuration_name=config.SEMANTIC_CONFIG,
        filter=_role_filter(role),
        select=["title", "chunk"],
        top=config.TOP_K,
    )
    chunks = [
        {"title": r["title"], "chunk": r["chunk"], "score": r.get("@search.reranker_score") or 0.0}
        for r in results
    ]
    if config.BASELINE:
        return chunks
    chunks = prefer_newest(search_query, chunks)
    strong = [c for c in chunks if c["score"] >= config.RERANKER_MIN_SCORE]
    return strong if strong else [c for c in chunks if c["score"] > 0]

def confidence_of(chunks: list[dict]) -> str:
    if not chunks:
        return "none"
    top = max(c["score"] for c in chunks)
    if top >= config.HIGH_CONFIDENCE_SCORE:
        return "high"
    if top >= config.RERANKER_MIN_SCORE:
        return "medium"
    return "low"

_cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()

def _cache_key(question: str, role: str | None) -> str:
    return f"{(role or 'all').lower()}|{' '.join(question.lower().split())}"

def _cache_get(key: str) -> dict | None:
    hit = _cache.get(key)
    if not hit:
        return None
    ts, value = hit
    if time.time() - ts > config.CACHE_TTL_SECONDS:
        _cache.pop(key, None)
        return None
    _cache.move_to_end(key)
    return value

def _cache_put(key: str, value: dict) -> None:
    _cache[key] = (time.time(), value)
    _cache.move_to_end(key)
    while len(_cache) > config.CACHE_MAX_ITEMS:
        _cache.popitem(last=False)

def build_messages(question: str, chunks: list[dict], history: list[dict]) -> list[dict]:
    context = "\n\n".join(f"[{c['title']}]\n{c['chunk']}" for c in chunks)
    return (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history[-config.HISTORY_MAX_TURNS :]
        + [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]
    )

def cited_sources(answer: str, chunks: list[dict]) -> list[str]:
    if config.REFUSAL_TEXT.lower() in answer.lower():
        return []
    retrieved = {c["title"] for c in chunks}
    cited = [t for t in re.findall(r"\[([^\]]+)\]", answer) if t in retrieved]
    return list(dict.fromkeys(cited))

def _refusal(search_query: str) -> dict:
    return {
        "answer": config.REFUSAL_TEXT,
        "sources": [],
        "query": search_query,
        "confidence": "none",
        "retrieved": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }

def answer(question: str, history: list[dict] | None = None,
           role: str | None = None) -> dict:
    history = history or []
    started = time.perf_counter()
    cache_key = _cache_key(question, role) if not history and not config.BASELINE else None
    if cache_key:
        cached = _cache_get(cache_key)
        if cached:
            log.info("cache hit role=%s question=%r", role, question)
            return {**cached, "cached": True}

    search_query = rewrite_query(question, history)
    chunks = retrieve(search_query, role)
    if not chunks and not config.BASELINE:
        return _refusal(search_query)

    resp = _chat.chat.completions.create(
        model=config.OPENAI_CHAT_DEPLOYMENT,
        messages=build_messages(question, chunks, history),
    )
    answer_text = resp.choices[0].message.content
    result = {
        "answer": answer_text,
        "sources": cited_sources(answer_text, chunks),
        "query": search_query,
        "confidence": confidence_of(chunks),
        "retrieved": [{"title": c["title"], "score": round(c["score"], 3)} for c in chunks],
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        },
    }
    if cache_key:
        _cache_put(cache_key, result)
    log.info(
        "answer role=%s confidence=%s top_score=%.2f tokens=%d+%d latency_ms=%d",
        role, result["confidence"],
        chunks[0]["score"] if chunks else 0.0,
        resp.usage.prompt_tokens, resp.usage.completion_tokens,
        int((time.perf_counter() - started) * 1000),
    )
    return result

def stream_answer(question: str, history: list[dict] | None = None,
                  role: str | None = None):
    history = history or []

    if not history:
        cached = _cache_get(_cache_key(question, role))
        if cached:
            yield _sse({"type": "retrieval", "chunks": cached["retrieved"],
                        "query": cached.get("query", question),
                        "confidence": cached.get("confidence", "unknown"),
                        "cached": True})
            yield _sse({"type": "text-delta", "text": cached["answer"]})
            yield _sse({"type": "sources", "items": cached["sources"]})
            yield _sse({"type": "usage", **cached["usage"], "cached": True})
            yield _sse({"type": "done"})
            return

    search_query = rewrite_query(question, history)
    chunks = retrieve(search_query, role)
    yield _sse({"type": "retrieval", "query": search_query,
                "confidence": confidence_of(chunks),
                "chunks": [
                    {"title": c["title"], "score": round(c["score"], 3)} for c in chunks
                ]})

    if not chunks:
        yield _sse({"type": "text-delta", "text": config.REFUSAL_TEXT})
        yield _sse({"type": "sources", "items": []})
        yield _sse({"type": "usage", "prompt_tokens": 0, "completion_tokens": 0})
        yield _sse({"type": "done"})
        return

    collected: list[str] = []
    usage = None
    started = time.perf_counter()
    stream = _chat.chat.completions.create(
        model=config.OPENAI_CHAT_DEPLOYMENT,
        messages=build_messages(question, chunks, history),
        stream=True,
        stream_options={"include_usage": True},
    )
    for event in stream:
        if event.choices and event.choices[0].delta.content:
            delta = event.choices[0].delta.content
            collected.append(delta)
            yield _sse({"type": "text-delta", "text": delta})
        if getattr(event, "usage", None):
            usage = {
                "prompt_tokens": event.usage.prompt_tokens,
                "completion_tokens": event.usage.completion_tokens,
            }
    full = "".join(collected)
    sources = cited_sources(full, chunks)
    yield _sse({"type": "sources", "items": sources})
    usage = usage or {"prompt_tokens": 0, "completion_tokens": 0}
    yield _sse({"type": "usage", **usage})
    yield _sse({"type": "done"})

    result = {
        "answer": full,
        "sources": sources,
        "query": search_query,
        "confidence": confidence_of(chunks),
        "retrieved": [{"title": c["title"], "score": round(c["score"], 3)} for c in chunks],
        "usage": usage,
    }
    _cache_put(_cache_key(question, role), result)
    log.info(
        "stream answer role=%s confidence=%s tokens=%d+%d latency_ms=%d",
        role, result["confidence"],
        usage["prompt_tokens"], usage["completion_tokens"],
        int((time.perf_counter() - started) * 1000),
    )

def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"
