# TAO-Fin Research Pipeline

This is the **research half** of your capstone project. It's a separate,
standalone Python package from your deployed Manus web app -- built to
close the specific gaps found when we read that app's actual source code:
no LLM reasoner, no semantic embeddings, a lexical-overlap "cross-encoder"
stand-in, and no evaluation harness at all.

**Keep the Manus app.** It's your System/Demo -- real SEC EDGAR retrieval,
a real deterministic controller, a working UI, screenshots and a live URL
for the paper's "Implementation" section. This package is what generates
your Methods and Results sections: real embeddings, a real cross-encoder,
a real LLM reasoner, and a benchmark comparing the TAO-controlled pipeline
against a naive single-pass baseline.

## What's real here vs. what needs your own machine

| Component | Status in this package |
|---|---|
| Calculator, Verifier gates, TAO controller | Real, deterministic, **32/32 tests pass in this sandbox**, no external dependencies |
| SEC EDGAR live filing-text retrieval | Real (`tao_fin/retrieval.py:fetch_filing_html`) -- fetches your ticker's actual 10-K/10-Q, no mocking |
| SEC EDGAR live XBRL numeric retrieval | Real (`tao_fin/xbrl.py:fetch_annual_figures`) -- auto-pulls revenue and operating income for the two most recent fiscal years, so numbers no longer have to be typed by hand, in either the deployed app or here |
| Fully automated ticker+question entry point | Real (`pipeline.run_fully_automated_analysis`) -- proven end-to-end in `tests/test_pipeline_automated.py` with mocked network calls; no manual revenue, operating income, or evidence excerpt required anywhere |
| BM25 + RRF fusion | Real, correct implementation, tested |
| Sentence-Transformers embeddings + cross-encoder reranker | **Real code, but untested here** -- this sandbox cannot reach `huggingface.co` to download the pretrained weights. Tests instead inject a fake embedder/cross-encoder to verify the *ranking logic* is correct; run this on a machine with normal internet access to prove the *real* models behave as expected. |
| LLM Reasoner (the component that was completely missing from the deployed app) | Real code calling the Anthropic API (`tao_fin/reasoner.py`) -- requires your own `ANTHROPIC_API_KEY` and will make real, billed calls |
| Evaluation harness (`eval/run_eval.py`) | Real metrics computation, **but the numbers currently printed are from a mocked reasoner** (smoke-test mode) -- see below |
| Threshold calibration (`eval/calibrate_threshold.py`) | Real precision/recall/F1 sweep logic, tested against synthetic scores -- needs your own labeled claim/passage pairs and the real cross-encoder to produce an actual number |

## Honest status of `eval/run_eval.py`'s current output

Running `python -m eval.run_eval` right now uses a small deterministic mock
in place of a real LLM call, because this sandbox has no API key and no
internet access to huggingface.co. That run is only proof the harness's
plumbing and metrics computation work end to end -- **do not put those
numbers in your paper.**

Worth knowing: that smoke test genuinely found and fixed a real bug. The
first version of the logic-consistency gate scanned an entire conclusion
sentence for contradiction words, which meant a *correct* explanation like
"revenue grew and expenses rose faster, compressing margin" got flagged as
a false contradiction, because "grew"/"rose" appeared in the same sentence
as "margin" even though they describe different things. It's now a
word-window check around each mention of "margin" specifically. This is a
legitimate example of an eval-driven bug fix worth mentioning in a
limitations/lessons-learned section -- and a sign the gate is still a
heuristic, not a parser; validate it against labeled examples before
trusting it fully.

## Running the real experiment

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...          # required for real reasoning calls
export SEC_EDGAR_USER_AGENT="YourName you@example.com"

python -m pytest tests/ -v               # 32 deterministic tests, no key needed
python -m eval.run_eval --real           # real Claude + real embeddings/cross-encoder
```

The first `--real` run will download the embedding and cross-encoder
models from Hugging Face (a few hundred MB, one-time). Make sure whatever
machine you run this on has normal internet access -- a personal laptop or
a cloud VM, not this sandbox.

To try the fully automated path directly (just a ticker and a question,
no manual numbers or evidence):

```python
from tao_fin.pipeline import run_fully_automated_analysis
from tao_fin.retrieval import default_embedder, default_cross_encoder
from tao_fin.reasoner import default_completion_fn

result = run_fully_automated_analysis(
    question="Why did operating margin change in the most recent fiscal year?",
    ticker="AAPL",
    user_agent="YourProject you@example.com",
    embedder=default_embedder(),
    cross_encoder=default_cross_encoder(),
    completion_fn=default_completion_fn(),
)
print(result.conclusion, result.release_status, result.confidence)
```

## Before you trust this at "10/10" for a journal submission

1. **Expand the benchmark.** `eval/benchmark.jsonl` has one real, verified
   case (Apple, matching the figures already validated in your deployed
   app's screenshots) plus three synthetic edge cases designed to exercise
   each controller path. A real evaluation needs on the order of dozens of
   real companies/questions with ground truth you've verified against the
   actual filings -- pull more real 10-Ks via `tao_fin/retrieval.py` and
   `tao_fin/xbrl.py` (both now fully automatic given just a ticker) and
   hand-verify the expected direction/magnitude for each.
2. **Tune the evidence-gate threshold empirically.** Run
   `python -m eval.calibrate_threshold --labels your_labels.jsonl` (see that
   file's docstring for the label format) against the real cross-encoder,
   then update `verifier.evidence_gate`'s default `threshold` to whatever
   the sweep recommends. The starting value in the code is a placeholder,
   not a validated constant.
3. **Consider replacing the logic gate's keyword heuristic entirely.**
   Windowing around "margin" fixed one false positive (see below); it will
   not catch every case a real parser or a second LLM-based consistency
   check would. For a journal-grade system, an LLM self-consistency check
   ("does this conclusion's stated direction match this computed number,
   yes/no") is more defensible than keyword proximity.
4. **Report both halves of the project honestly.** The Manus app
   demonstrates a real, deployed, user-facing controlled-release system.
   This package demonstrates the actual research contribution -- verified,
   test-covered, real hybrid retrieval, real automatic XBRL + filing-text
   sourcing, and real controller branching -- but its reasoning-quality
   numbers don't exist until you run `--real` on a machine with API access.
   Don't report smoke-test numbers as results.

## Project layout

```
tao_fin/
  calculator.py     deterministic margin arithmetic (no LLM involved, ever)
  query_analyzer.py free-text question -> difficulty + task type
  retrieval.py       SEC EDGAR filing-text fetch + chunk + FAISS/BM25/RRF/cross-encoder
  xbrl.py            SEC EDGAR XBRL numeric fetch -- auto revenue/operating income
  reasoner.py        real Anthropic API call -> grounded causal claims (JSON)
  verifier.py        four gates: calculations, evidence, logic, sanity
  controller.py      STOP/REVISE/RETRIEVE/RECALCULATE branching + confidence
  pipeline.py        run_controlled_analysis (manual/hybrid) +
                      run_fully_automated_analysis (ticker + question only)
tests/               32 tests, all deterministic, no network/API key needed
eval/
  benchmark.jsonl         1 real verified case + 3 labeled synthetic edge cases
  run_eval.py             baseline vs. TAO comparison harness (--real for actual results)
  calibrate_threshold.py  empirically tunes the evidence-gate threshold
```
