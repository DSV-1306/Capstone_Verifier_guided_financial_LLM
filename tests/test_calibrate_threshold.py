import numpy as np

from eval.calibrate_threshold import sweep_thresholds


def test_sweep_finds_perfect_separator_when_scores_cleanly_separate():
    # grounded claims score high, ungrounded score low -- a clean separator exists
    scores = np.array([0.9, 0.85, 0.8, 0.2, 0.1, 0.05])
    labels = np.array([True, True, True, False, False, False])
    results = sweep_thresholds(scores, labels, n_steps=50)
    best = max(results, key=lambda r: r["f1"])
    assert best["f1"] == 1.0
    assert 0.2 < best["threshold"] <= 0.8


def test_sweep_reports_imperfect_f1_on_overlapping_scores():
    scores = np.array([0.9, 0.6, 0.55, 0.5, 0.4, 0.1])
    labels = np.array([True, True, False, True, False, False])
    results = sweep_thresholds(scores, labels, n_steps=50)
    best = max(results, key=lambda r: r["f1"])
    assert best["f1"] < 1.0  # no threshold perfectly separates this overlapping data
