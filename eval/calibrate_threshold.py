"""
Empirically calibrates the evidence-gate's cross-encoder threshold, instead
of trusting the starting-point constant in verifier.evidence_gate.

This is the tool for the open item flagged in the README: "tune the
evidence-gate threshold empirically." It needs a labeled set of
(claim, passage, is_grounded) triples and the REAL cross-encoder -- run it
on your own machine after collecting labels, not in this sandbox.

Usage:
    python -m eval.calibrate_threshold --labels eval/grounding_labels.jsonl

Each line of the labels file should look like:
    {"claim": "...", "passage": "...", "is_grounded": true}

Label these by hand: take real reasoner outputs from a `--real` eval run,
and for each claim, judge yourself whether the cited passage actually
supports it. Aim for at least ~50-100 labeled pairs, with a mix of clearly
grounded, clearly ungrounded, and genuinely borderline cases -- borderline
cases are what actually determines where the threshold should sit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_labels(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sweep_thresholds(scores: np.ndarray, labels: np.ndarray, n_steps: int = 200) -> list[dict]:
    """Try thresholds spanning the observed score range and report precision/
    recall/F1 for each, so you can pick a threshold matching what your paper
    cares about more -- e.g. high precision (never falsely call something
    grounded) vs. high recall (never falsely reject a genuinely grounded claim)."""
    lo, hi = float(scores.min()), float(scores.max())
    rows = []
    for t in np.linspace(lo, hi, n_steps):
        predicted_grounded = scores >= t
        tp = int(np.sum(predicted_grounded & labels))
        fp = int(np.sum(predicted_grounded & ~labels))
        fn = int(np.sum(~predicted_grounded & labels))
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({"threshold": round(float(t), 4), "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()

    from tao_fin.retrieval import default_cross_encoder  # real model -- needs internet access

    cross_encoder = default_cross_encoder()
    labeled = load_labels(args.labels)
    scores = cross_encoder([(item["claim"], item["passage"]) for item in labeled])
    labels = np.array([item["is_grounded"] for item in labeled])

    results = sweep_thresholds(np.asarray(scores), labels)
    best = max(results, key=lambda r: r["f1"])
    print(f"Best F1 threshold: {best['threshold']} (precision={best['precision']}, recall={best['recall']}, f1={best['f1']})")
    print("\nNearby thresholds for context:")
    for r in results:
        if abs(r["threshold"] - best["threshold"]) < (scores.max() - scores.min()) * 0.05:
            print(r)
    print(
        f"\nUpdate verifier.evidence_gate's default `threshold` parameter to {best['threshold']}, "
        "then rerun tests/ and eval/run_eval.py --real to confirm nothing regressed."
    )


if __name__ == "__main__":
    main()
