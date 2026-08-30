"""
Retrieval pipeline: SEC EDGAR -> parse -> financial-aware chunk -> hybrid
FAISS (semantic) + BM25 (lexical) retrieval -> RRF fusion -> cross-encoder
rerank -> top-k evidence passages.

This is the part your deployed Manus app approximates with a hashed
bag-of-words vector and a term-overlap "cross-encoder" fallback. Here the
embedder and reranker are real pretrained models (Sentence-Transformers /
a real cross-encoder), injected as dependencies so the ranking logic itself
stays unit-testable without needing model downloads or network access.

NOTE ON RUNNING THIS FOR REAL: `SentenceTransformer` and `CrossEncoder`
download pretrained weights from huggingface.co on first use. That domain
is not reachable from this sandbox, so real model downloads must happen on
your own machine / server with normal internet access -- the code is written
to do the right thing there. Tests in tests/ inject a fake embedder so the
ranking math itself is verified without network access.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import requests
from rank_bm25 import BM25Okapi

SEC_BASE = "https://www.sec.gov"
SEC_DATA = "https://data.sec.gov"

EmbedderFn = Callable[[Sequence[str]], np.ndarray]
CrossEncoderFn = Callable[[Sequence[tuple[str, str]]], np.ndarray]


@dataclass
class Passage:
    id: str
    excerpt: str


@dataclass
class RankedPassage:
    id: str
    excerpt: str
    bm25_rank: int
    vector_rank: int
    rrf_score: float
    rerank_score: float


def default_embedder() -> EmbedderFn:
    """Real Sentence-Transformers embedder. Requires network access to
    huggingface.co on first call to download the model -- run this on a
    machine with normal internet access, not this sandbox."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed(texts: Sequence[str]) -> np.ndarray:
        return model.encode(list(texts), normalize_embeddings=True)

    return embed


def default_cross_encoder() -> CrossEncoderFn:
    """Real cross-encoder reranker (MS MARCO MiniLM), not a lexical-overlap
    stand-in. Same network caveat as default_embedder."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def score(pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        return model.predict(list(pairs))

    return score


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def rank_passages_hybrid(
    passages: Sequence[Passage],
    query: str,
    embedder: EmbedderFn,
    cross_encoder: CrossEncoderFn,
    rrf_k: int = 60,
    top_k: int = 5,
) -> list[RankedPassage]:
    """Real hybrid retrieval: BM25 + semantic vector search, fused with
    Reciprocal Rank Fusion, then reranked with a cross-encoder.

    This is the actual research-grade version of your architecture's
    'FAISS + BM25 -> RRF -> Cross-Encoder -> Top Evidence' pipeline.
    """
    if not passages:
        return []

    corpus_tokens = [_tokenize(p.excerpt) for p in passages]
    bm25 = BM25Okapi(corpus_tokens)
    bm25_scores = bm25.get_scores(_tokenize(query))
    bm25_order = np.argsort(-bm25_scores)
    bm25_rank = {passages[i].id: int(rank) + 1 for rank, i in enumerate(bm25_order)}

    doc_vectors = embedder([p.excerpt for p in passages])
    query_vector = embedder([query])[0]
    similarities = doc_vectors @ query_vector
    vector_order = np.argsort(-similarities)
    vector_rank = {passages[i].id: int(rank) + 1 for rank, i in enumerate(vector_order)}

    rrf_scores = {
        p.id: 1.0 / (rrf_k + bm25_rank[p.id]) + 1.0 / (rrf_k + vector_rank[p.id])
        for p in passages
    }

    fused_order = sorted(passages, key=lambda p: -rrf_scores[p.id])[: max(top_k * 3, top_k)]
    rerank_scores = cross_encoder([(query, p.excerpt) for p in fused_order])

    ranked = [
        RankedPassage(
            id=p.id,
            excerpt=p.excerpt,
            bm25_rank=bm25_rank[p.id],
            vector_rank=vector_rank[p.id],
            rrf_score=rrf_scores[p.id],
            rerank_score=float(score),
        )
        for p, score in zip(fused_order, rerank_scores)
    ]
    ranked.sort(key=lambda r: -r.rerank_score)
    return ranked[:top_k]


def _strip_html(raw: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", raw, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def chunk_filing(raw_html: str, min_len: int = 120, max_len: int = 4000) -> list[Passage]:
    """Financial-aware chunking: keep each <p>/<div>/<table> block intact
    rather than splitting mid-sentence or mid-table."""
    blocks = re.findall(r"<(p|div|table)\b[^>]*>([\s\S]*?)</\1>", raw_html, flags=re.I)
    seen = set()
    passages: list[Passage] = []
    for _, inner in blocks:
        text = _strip_html(inner)
        if min_len <= len(text) <= max_len and text not in seen:
            seen.add(text)
            passages.append(Passage(id=f"p{len(passages) + 1}", excerpt=text))
    return passages


def fetch_filing_identity(ticker: str, user_agent: str) -> dict:
    """Resolve a ticker to its most recent 10-K/10-Q via SEC EDGAR (real, live)."""
    headers = {"User-Agent": user_agent}
    index = requests.get(f"{SEC_DATA}/files/company_tickers.json", headers=headers, timeout=20)
    index.raise_for_status()
    entry = next(
        (v for v in index.json().values() if v["ticker"].upper() == ticker.upper()), None
    )
    if entry is None:
        raise ValueError(f"No SEC registrant found for ticker {ticker!r}.")
    cik = str(entry["cik_str"]).zfill(10)
    submissions = requests.get(f"{SEC_DATA}/submissions/CIK{cik}.json", headers=headers, timeout=20)
    submissions.raise_for_status()
    recent = submissions.json()["filings"]["recent"]
    idx = next(
        (i for i, form in enumerate(recent["form"]) if form in ("10-K", "10-Q")), None
    )
    if idx is None:
        raise ValueError(f"No recent 10-K/10-Q found for {ticker!r}.")
    accession = recent["accessionNumber"][idx].replace("-", "")
    doc = recent["primaryDocument"][idx]
    return {
        "ticker": ticker.upper(),
        "form": recent["form"][idx],
        "filing_date": recent["filingDate"][idx],
        "url": f"{SEC_BASE}/Archives/edgar/data/{entry['cik_str']}/{accession}/{doc}",
    }


def fetch_filing_html(ticker: str, user_agent: str) -> tuple[str, dict]:
    identity = fetch_filing_identity(ticker, user_agent)
    response = requests.get(identity["url"], headers={"User-Agent": user_agent}, timeout=30)
    response.raise_for_status()
    return response.text, identity
