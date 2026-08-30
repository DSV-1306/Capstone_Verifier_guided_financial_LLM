"""
Fakes used across tests so the pipeline's *logic* (branching, gate
evaluation, confidence scoring) is verified without needing network access,
a Hugging Face model download, or a paid Anthropic API call.

None of these fakes are used in tao_fin/ itself -- production code always
calls the real embedder/cross-encoder/LLM. They exist only here, in tests/.
"""
from __future__ import annotations

import re

import numpy as np
import pytest


_STOPWORDS = {
    "a", "an", "and", "the", "of", "to", "in", "due", "this", "year", "fiscal",
    "for", "by", "company", "margin", "operating", "income", "expanded",
    "increased", "improved", "declined", "decreased",
}
_HASH_DIM = 256


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _content_tokens(text: str) -> set[str]:
    """Tokens with generic financial-reasoning boilerplate removed, so
    overlap scoring reflects actual shared substance rather than every
    claim trivially sharing words like 'margin' or 'expanded'."""
    return _tokenize(text) - _STOPWORDS


@pytest.fixture
def fake_embedder():
    """Deterministic fixed-dimension hashed bag-of-words embedder. Fixed
    dimension (unlike a growing vocab) so it's safe to call separately for
    documents and for a query, which is exactly how retrieval.py calls it."""

    def embed(texts):
        vectors = np.zeros((len(texts), _HASH_DIM))
        for row, text in enumerate(texts):
            for token in _tokenize(text):
                vectors[row, hash(token) % _HASH_DIM] += 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms

    return embed


@pytest.fixture
def fake_cross_encoder():
    """Content-word overlap score in [0, 1], with generic boilerplate terms
    excluded so two claims that only share words like 'margin' or 'expanded'
    don't falsely register as grounded in each other. Deliberately simple so
    test expectations are easy to reason about -- the real cross-encoder in
    tao_fin/retrieval.py replaces this in production."""

    def score(pairs):
        scores = []
        for query, excerpt in pairs:
            q, e = _content_tokens(query), _content_tokens(excerpt)
            scores.append(len(q & e) / max(len(q), 1))
        return np.array(scores)

    return score


def scripted_completion_fn(responses: list[str]):
    """Returns a completion_fn that yields each response in `responses` in
    order, repeating the last one if called more times than provided."""
    calls = {"n": 0}

    def complete(system: str, user: str) -> str:
        i = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        return responses[i]

    return complete


@pytest.fixture
def make_scripted_completion_fn():
    """Fixture wrapper around scripted_completion_fn, so test files receive
    it via pytest's fixture injection instead of a direct
    `from tests.conftest import ...`. Direct imports of the tests package
    are fragile on machines where some other installed package also ships a
    top-level `tests/` folder (a real namespace-package collision found by
    actually running this on Windows) -- fixture injection sidesteps that
    entirely, since pytest discovers conftest.py through its own plugin
    mechanism rather than a plain Python import.
    """
    return scripted_completion_fn
