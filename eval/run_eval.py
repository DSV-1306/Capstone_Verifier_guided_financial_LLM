"""
Evaluation harness: baseline (naive single-pass, no verifier) vs the full
TAO-controlled pipeline, over eval/benchmark.jsonl.

*** IMPORTANT: running this file as-is uses a MOCKED reasoner (see
`smoke_test_completion_fn` below), because this sandbox cannot reach
huggingface.co or make paid Anthropic calls. It exists to prove the harness's
plumbing and metrics are correct -- it is NOT a real experimental result and
must not be reported as one in the paper.

TO RUN THIS FOR REAL (on your own machine, with internet + an API key):

    export ANTHROPIC_API_KEY=sk-...
    python -m eval.run_eval --real

which swaps in `reasoner.default_completion_fn()`,
`retrieval.default_embedder()`, and `retrieval.default_cross_encoder()` --
real Claude reasoning, real Sentence-Transformers embeddings, real
cross-encoder reranking. That combination is what should generate the
numbers in your Results section.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tao_fin.calculator import compute_margin_ledger
from tao_fin.pipeline import run_controlled_analysis
from tao_fin.reasoner import Claim, ReasoningResult, reason
from tao_fin.retrieval import Passage
from tao_fin.verifier import evidence_gate

BENCHMARK_PATH = Path(__file__).parent / "benchmark.jsonl"


def load_benchmark() -> list[dict]:
    return [json.loads(line) for line in BENCHMARK_PATH.read_text().splitlines() if line.strip()]


def smoke_test_completion_fn():
    """Deterministic mock standing in for a real LLM call, ONLY so this
    script can run without network access or an API key. Roughly half the
    time it invents a plausible-sounding but ungrounded causal driver --
    on purpose, to prove the verifier actually catches it. Do not mistake
    this for a real reasoning model's behavior."""
    call_count = {"n": 0}
    fabricated_drivers = [
        "a one-time tax settlement",
        "an unannounced pricing change",
        "favorable currency hedging gains",
    ]

    def complete(system: str, user: str) -> str:
        call_count["n"] += 1
        evidence_block = user.split("Retrieved evidence passages:")[-1]
        question = user.split("\n")[0]
        # crude "does the evidence contain causal language" heuristic, purely
        # so the mock behaves differently across the benchmark's designed cases
        has_causal_language = bool(
            re.search(r"driven by|due to|compressing|reported no material change|flat year over year", evidence_block, re.I)
        )
        if has_causal_language or call_count["n"] > 1:
            # second attempt onward: "retrieved more" or "revised" -> ground it properly
            sentence = re.split(r"(?<=[.])\s", evidence_block.strip())[0]
            return json.dumps({
                "conclusion": f"{sentence.strip()}",
                "claims": [{"text": sentence.strip(), "supporting_passage_id": "p1"}],
            })
        driver = fabricated_drivers[call_count["n"] % len(fabricated_drivers)]
        return json.dumps({
            "conclusion": f"The change was primarily driven by {driver}.",
            "claims": [{"text": f"{driver} drove the change", "supporting_passage_id": "p1"}],
        })

    return complete


def naive_baseline(item: dict, embedder, cross_encoder, completion_fn) -> dict:
    """No verifier, no controller loop -- calls the reasoner exactly once and
    reports whatever it says, the way a plain 'LLM answers a financial
    question' baseline would. We still run the SAME evidence gate against its
    output afterward (not to gate release -- baseline never checks -- but to
    measure how often an unguarded system would have shipped an ungrounded
    causal claim as a confident answer)."""
    ledger = compute_margin_ledger(
        item["current_revenue"], item["current_operating_income"],
        item["prior_revenue"], item["prior_operating_income"],
    )
    passages = [Passage(id="p1", excerpt=item["evidence_excerpt"])]
    ledger_summary = "\n".join(f"- {li.label}: {li.formula} = {li.result}" for li in ledger)
    result = reason(item["question"], item["ticker"], passages, ledger_summary, completion_fn)
    would_pass_evidence_gate = evidence_gate(result, passages, cross_encoder).passed
    return {"conclusion": result.conclusion, "grounded": would_pass_evidence_gate}


def direction_word(movement: float, epsilon: float = 0.015) -> str:
    if movement > epsilon:
        return "expand"
    if movement < -epsilon:
        return "contract"
    return "unchanged"


def run(real: bool) -> list[dict]:
    if real:
        from tao_fin.reasoner import default_completion_fn
        from tao_fin.retrieval import default_cross_encoder, default_embedder
        embedder, cross_encoder, completion_fn = default_embedder(), default_cross_encoder(), default_completion_fn()
    else:
        print("*** SMOKE TEST MODE: mocked reasoner, not a real experiment. See module docstring. ***\n")
        # Tiny inline fakes -- kept self-contained here (not imported from
        # tests/) so `eval/` never depends on the tests package at all.
        import numpy as np
        def embedder(texts):
            vecs = np.zeros((len(texts), 256))
            for row, t in enumerate(texts):
                for tok in re.findall(r"[a-z0-9]+", t.lower()):
                    vecs[row, hash(tok) % 256] += 1
            norms = np.linalg.norm(vecs, axis=1, keepdims=True); norms[norms == 0] = 1
            return vecs / norms
        stop = {"a","an","and","the","of","to","in","due","this","year","fiscal","for","by","company","margin","operating","income","expanded","increased","improved","declined","decreased"}
        def cross_encoder(pairs):
            scores = []
            for q, e in pairs:
                qt = set(re.findall(r"[a-z0-9]+", q.lower())) - stop
                et = set(re.findall(r"[a-z0-9]+", e.lower())) - stop
                scores.append(len(qt & et) / max(len(qt), 1))
            return np.array(scores)
        completion_fn = smoke_test_completion_fn()

    rows = []
    for item in load_benchmark():
        def retrieve_fn(expanded: bool, item=item):
            return [Passage(id="p1", excerpt=item["evidence_excerpt"])]

        baseline = naive_baseline(item, embedder, cross_encoder, completion_fn if real else smoke_test_completion_fn())
        tao_result = run_controlled_analysis(
            question=item["question"], ticker=item["ticker"],
            current_revenue=item["current_revenue"], current_operating_income=item["current_operating_income"],
            prior_revenue=item["prior_revenue"], prior_operating_income=item["prior_operating_income"],
            retrieve_fn=retrieve_fn, embedder=embedder, cross_encoder=cross_encoder,
            completion_fn=completion_fn if real else smoke_test_completion_fn(),
        )
        expected_dir = item["expected_direction"]
        tao_dir = direction_word(tao_result.ledger[2].result)
        rows.append({
            "id": item["id"],
            "is_real": item["is_real"],
            "expected_direction": expected_dir,
            "tao_direction_correct": tao_dir == expected_dir,
            "tao_release_status": tao_result.release_status,
            "tao_iterations": len(tao_result.trace),
            "tao_compute_saved_pct": tao_result.compute_saved_percent,
            "tao_confidence": tao_result.confidence,
            "baseline_grounded": baseline["grounded"],
            "baseline_would_ship_ungrounded_claim": not baseline["grounded"],
        })
    return rows


def print_table(rows: list[dict]) -> None:
    header = ["id", "expected_dir", "TAO_correct", "TAO_status", "TAO_iters", "compute_saved%", "confidence", "baseline_grounded"]
    print(" | ".join(header))
    print(" | ".join("-" * len(h) for h in header))
    for r in rows:
        print(" | ".join(str(v) for v in [
            r["id"], r["expected_direction"], r["tao_direction_correct"], r["tao_release_status"],
            r["tao_iterations"], r["tao_compute_saved_pct"], r["tao_confidence"], r["baseline_grounded"],
        ]))
    n = len(rows)
    baseline_ungrounded_rate = sum(r["baseline_would_ship_ungrounded_claim"] for r in rows) / n
    tao_accuracy = sum(r["tao_direction_correct"] for r in rows) / n
    avg_compute_saved = sum(r["tao_compute_saved_pct"] for r in rows) / n
    print(f"\nBaseline ungrounded-claim rate (would have shipped as-is): {baseline_ungrounded_rate:.0%}")
    print(f"TAO direction accuracy (grounded, verified conclusions): {tao_accuracy:.0%}")
    print(f"TAO average compute saved vs. fixed 3-iteration budget: {avg_compute_saved:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Use real Claude reasoning + real embeddings/cross-encoder (requires ANTHROPIC_API_KEY and internet access to huggingface.co)")
    args = parser.parse_args()
    print_table(run(real=args.real))
