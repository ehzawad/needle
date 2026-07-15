# File-defined-scope bot + a miniature MLOps pipeline

A conversational bot whose **domain lives in an editable file** ([`seed16/cards.json`](seed16/cards.json)),
built on the **frozen** base Qwen3-4B-Instruct (no fine-tuning). A per-query LLM gate,
backstopped by a **deterministic fail-closed policy**, decides ANSWER / CLARIFY / ABSTAIN
*before* generating; answers are grounded only in a card's approved facts. On a 50-scenario
dev set it gets **0/25 harmful leaks · 20/20 right-card in-scope · 5/5 ambiguous→clarify**
(a dev gate, not a certification — see `RAG_SCOPE.md`).

Around it sits [`pipeline/`](pipeline/) — the *automation layer* a real ML org would run
(registry, DAG, canary promotion, observability, A5000-only GPU isolation), as a learning
artifact. Deliberately overkill for 16 cards.

## Quick start

```bash
cd examples/qwen_sft
bash setup_env.sh          # one-time: pinned inference venv (.venv-qlora)

# talk to the bot directly (loads Qwen on the A5000 — see the A5000 note in setup_env.sh)
.venv-qlora/bin/python scope_bot.py --interactive

# the full MLOps pipeline, GPU-free (mock gate) — the CI path
.venv-qlora/bin/python -m unittest discover -s pipeline/tests -q
.venv-qlora/bin/python -m pipeline run --config pipeline/config.ci.json --state .pipeline-ci --backend mock --promote --actor ci

# the pipeline on the real model (A5000-pinned by the guard; no CUDA env needed)
.venv-qlora/bin/python -m pipeline run --config pipeline/config.demo.json --state .pipeline-a5000 --backend real --promote --actor demo

# serve the promoted model over HTTP (real: guards onto the A5000, closes the feedback loop)
.venv-qlora/bin/python -m pipeline serve --config pipeline/config.demo.json --state .pipeline-a5000 --http 127.0.0.1:8080 --backend real
curl -s :8080/respond -d '{"query":"How do I add my card to Apple Pay?"}'
```

## File map

| Path | Purpose |
|---|---|
| `scope_bot.py` | the bot: frozen Qwen gate + grounded responder + opt-in exemplar injection; `--query`/`--interactive` |
| `scope_policy.py` | deterministic, downgrade-only discriminator enforcement (the safety floor) |
| `seed16/cards.json` | the editable scope/knowledge file (16 cards; 6 carry executable discriminators) |
| `eval50.py` · `eval50_results.txt` | the frozen 50-scenario dev suite + its committed baseline output |
| `feedback_log.py` · `mine_signals.py` · `adapter/learn.py` | the feedback loop: log turns → mine weak labels → build the exemplar bank (libraries the pipeline wraps) |
| `logs/conversations.sample.jsonl` | illustrative traffic + CI/demo pipeline input |
| `pipeline/` | the MLOps automation package (see [`pipeline/PIPELINE.md`](pipeline/PIPELINE.md)) |
| `setup_env.sh` | pinned inference venv bootstrap |

## Docs

- [`RAG_SCOPE.md`](RAG_SCOPE.md) — architecture decision: why file-defined scope, gate-before-generation, honest limits, why no domain SFT.
- [`FEEDBACK_PIPELINE.md`](FEEDBACK_PIPELINE.md) — the feedback-signal taxonomy and exemplar-bank semantics.
- [`pipeline/PIPELINE.md`](pipeline/PIPELINE.md) — the automation runbook: DAG, registry, release channels, gates, device isolation, and each component's real-world analog.

## ⚠️ Freeze

`pipeline/config.{ci,demo}.json` pin the SHA-256 of `scope_bot.py`, `scope_policy.py`, and
`eval50.py`. Editing any of those three **fails pipeline ingest** unless you re-freeze both
config hashes in the same change. Cards may change freely → they produce a new candidate.
Runtime state (`.pipeline-*/`, `logs/*.jsonl` real logs, `exemplars.json`) is git-ignored.
