from tao_fin.retrieval import Passage, chunk_filing, rank_passages_hybrid


PASSAGES = [
    Passage(id="a", excerpt="Operating margin expanded due to lower freight and component costs this year."),
    Passage(id="b", excerpt="The board approved a new executive compensation plan for fiscal year 2025."),
    Passage(id="c", excerpt="Gross margin improved as component costs declined and freight expenses fell."),
]


def test_hybrid_ranking_surfaces_relevant_passage_first(fake_embedder, fake_cross_encoder):
    ranked = rank_passages_hybrid(
        PASSAGES,
        query="why did operating margin improve due to lower costs",
        embedder=fake_embedder,
        cross_encoder=fake_cross_encoder,
        top_k=2,
    )
    ids = [r.id for r in ranked]
    assert "b" not in ids  # the unrelated compensation-plan passage should not surface
    assert ids[0] in {"a", "c"}


def test_empty_passages_returns_empty(fake_embedder, fake_cross_encoder):
    assert rank_passages_hybrid([], "any query", fake_embedder, fake_cross_encoder) == []


def test_chunking_keeps_tables_and_paragraphs_intact_and_drops_short_fragments():
    html = """
    <div><p>Total net sales were $416,161 million for fiscal 2025, compared with $391,035 million for fiscal 2024.</p></div>
    <div><p>ok</p></div>
    <table><tr><td>Operating income</td><td>133,050</td></tr></table>
    """
    chunks = chunk_filing(html, min_len=20, max_len=4000)
    texts = [c.excerpt for c in chunks]
    assert any("416,161" in t for t in texts)
    assert not any(t == "ok" for t in texts)  # too short, correctly dropped
