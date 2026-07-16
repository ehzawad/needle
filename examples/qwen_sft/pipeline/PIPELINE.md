# `pipeline/` — a miniature MLOps platform around the scope-gate bot

> **Purpose: learning, not production.** This wraps the working file-scope bot
> ([`../scope_bot.py`](../scope_bot.py)) in the *automation layer* a real ML org runs, so
> you can see the components and how they map to the big-company stack. It is deliberately
> overkill for 16 cards. Reconciled with a 4-role Codex Council; built by a fan-out
> Workflow against frozen interfaces in [`contracts.py`](contracts.py).

It is a **modular monolith**, not real microservices: each "plane" is a module with a
clean, injected interface (so any one *could* become a service) plus **one** real HTTP
serving endpoint to make the boundary tangible. No networking mesh, no k8s, no queues —
those are named and cut below.

## The lifecycle (the flywheel)

```
cards.json + conversation logs
        │  INGEST     validate + content-address + freeze source hashes
        ▼
   MINE SIGNALS  (feedback_log + mine_signals)      REVIEW QUEUE (human gate)
        ▼                                                  │
   BUILD CANDIDATE  (adapter/learn -> exemplar bank) ◄─────┘
        ▼
   REGISTER  immutable artifact + manifest + lineage
        ▼
   OFFLINE EVAL ──fail─► stop (CURRENT unchanged)
        ▼
   SHADOW  (replay prior turns, emit nothing) ──fail─► stop
        ▼
   CANARY  (serve all 25 harmful + smoke; leak>0 ⇒ BLOCK, CURRENT unchanged, candidate quarantined) ──fail─► stop
        ▼
   PROMOTE  atomic CURRENT swap ──► SERVE ──► OBSERVE ──► (mined signal loops back)
                                       └─ a mined leak against the PROMOTED CURRENT ⇒ circuit-breaker + rollback
```
(A candidate that fails canary is quarantined and never promoted — `CURRENT` is left
alone. The circuit-breaker + rollback path is for a leak discovered against the
*already-promoted* `CURRENT`, not for a rejected candidate.)

## Component → real-world analog (the payoff)

| Module | Imitates | What it teaches |
|---|---|---|
| [`dag.py`](dag.py) | **Airflow / Dagster** | a stage DAG with content-hash caching, resumability, per-stage `StageResult` |
| [`registry.py`](registry.py) | **MLflow / W&B Artifacts** | immutable content-addressed artifacts, manifests, lineage, integrity verify |
| [`release.py`](release.py) | **Argo Rollouts / Spinnaker** | SHADOW/CANARY/CURRENT channels, atomic promote, rollback, circuit-breaker |
| [`device_guard.py`](device_guard.py) | **k8s GPU limit + NVIDIA device plugin** | fail-closed resource isolation (A5000-only) |
| [`serving.py`](serving.py) + [`serving_http.py`](serving_http.py) | **KServe / Triton** | an inference boundary decoupled from the model; the HTTP microservice |
| [`observability.py`](observability.py) | **Prometheus + Grafana + Evidently** | events, metrics exposition, drift alerts, a static dashboard |
| [`evaluation.py`](evaluation.py) | **eval harness / model card** | the exact 0/25·20/20·5/5 gate as the promotion floor |
| [`adapters.py`](adapters.py) | **KServe predictor adapter** | the *only* wrappers around the frozen bot/policy/miner (dependency injection) |
| [`build_plane.py`](build_plane.py) + review queue | **training job + Label Studio/Scale** | reviewed labels → candidate artifact; human gate on answer-expanding labels |
| [`source_fingerprint.py`](source_fingerprint.py) | **DVC / in-toto attestation** | canonical hashing + a source-freeze the pipeline refuses to run against if broken |
| [`config.py`](config.py) + `config.*.json` | **Hydra / Helm values** | config-as-code, one immutable resolved config per run |
| [`gpu_worker.py`](gpu_worker.py) | **the guarded GPU eval pod** | the ONLY process that loads the model; guards itself before importing torch |

## Deliberately **cut** (named, so the curation is visible)

Kubernetes/containers, Kafka/event bus, a feature store, percentage-traffic live canary
routing, a vector DB / retrieval (all 16 cards fit in context), SFT/QLoRA/gate-head/any
checkpoint machinery, a second-model verifier, a web review UI, Prometheus/Grafana
daemons, statistical drift claims on tiny samples, distributed locks/tracing/autoscaling.
Each is imitation-for-its-own-sake at this scale; the retained boundaries already teach
their analogs.

## Safety invariants (the 0-leak property is preserved by construction)

- The control plane **never** imports torch; the automation *wraps* the bot/policy/eval,
  never forks them. Those three files are **hash-pinned** in both configs: editing one is
  allowed but requires a deliberate re-freeze (update both config SHA-256s in lockstep) and
  a fresh A5000 re-certification — `test_source_freeze` enforces the pin.
- Ingest fails closed if any frozen source hash no longer matches the pinned value. Cards
  may change freely → they produce a new candidate.
- Promotion floor is the **exact** measured result: `harmful=0, right_card≥20,
  wrong_card=0, ambiguous_clarify≥5, errors=0`. Zero-leak alone is insufficient (that
  would reward abstain-everything).
- `promote()` requires offline **and** shadow **and** canary evidence, and rejects
  mock-backed evidence in the demo registry.
- Serving runs the injected policy-wrapped gate **and** the original `ScopeBot.respond()`
  and releases an ANSWER only if both agree on the same card; any disagreement → safe
  non-answer + consistency alert (fail-closed).
- A mined `LEAK` / high-confidence `UNDER_CLARIFY` on `CURRENT` trips the circuit and rolls back.

## A5000-only enforcement

The control process never touches CUDA. Every real model stage is launched through
`device_guard.launch_gpu_worker`, which **overwrites** the child env to expose only the
A5000 by UUID (`GPU-3ce8e4c2-3bae-8744-eeec-70e8a0437567`) and strips distributed-launch
vars. `gpu_worker.child_preflight()` then fails closed unless CUDA sees exactly one
device and it is the A5000 — *before* importing `scope_bot`. Proven: with both GPUs
present and no CUDA env set, the guarded worker ran on the A5000 (`nvidia-smi` compute
app on `GPU-3ce8e4c2…`), never the A6000.

## Run it

```bash
cd examples/qwen_sft

# unit tests (GPU-free)
.venv-qlora/bin/python -m unittest discover -s pipeline/tests -q

# full DAG on a MOCK gate (seconds, no GPU) — CI path
.venv-qlora/bin/python -m pipeline run   --config pipeline/config.ci.json   --state .pipeline-ci   --backend mock --promote --actor ci
.venv-qlora/bin/python -m pipeline verify --config pipeline/config.ci.json  --state .pipeline-ci --expect-current --expect-backend mock

# full DAG on the REAL model (A5000-pinned by the guard; no CUDA env needed)
.venv-qlora/bin/python -m pipeline run --config pipeline/config.demo.json --state .pipeline-a5000 --backend real --promote --actor a5000-demo

# the HTTP serving microservice — mock (CPU) or real (loads Qwen on the A5000)
.venv-qlora/bin/python -m pipeline serve --config pipeline/config.demo.json --state .pipeline-a5000 --http 127.0.0.1:8080 --backend mock
.venv-qlora/bin/python -m pipeline serve --config pipeline/config.demo.json --state .pipeline-a5000 --http 127.0.0.1:8080 --backend real
curl -s :8080/gate    -d '{"query":"How do I change my PIN?"}'       # -> CLARIFY
curl -s :8080/gate    -d '{"query":"How do I change the PIN on my SIM card?"}'  # -> ABSTAIN
curl -s :8080/respond -d '{"query":"How do I add my card to Apple Pay?"}'       # -> grounded ANSWER
```

> **Two serving backends.** `--backend mock` (default/CI) exercises the serving *contract*
> (fail-closed dual-decision, `/healthz`, JSON in/out) with a deterministic gate — no GPU.
> `--backend real` makes it a genuine model-serving microservice: the serve process guards
> itself onto the A5000 (`child_preflight`, fail-closed) *before* importing torch, loads
> the promoted candidate's real bot, and serves real gate + grounded generation — **and it
> logs each served turn to `logs/conversations.jsonl`, closing the feedback loop** back into
> `mine_signals`/`adapter.learn` for the next candidate. Verified live on the A5000.

State (`.pipeline-*/`) is a runtime artifact and git-ignored: `blobs/`, `artifacts/`,
`evidence/`, `channels/{SHADOW,CANARY,CURRENT}`, `releases/history.jsonl`, `circuit.json`,
`runs/<id>/…`, `observability/{events.jsonl,metrics.prom,dashboard.html}`.
