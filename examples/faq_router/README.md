# FAQ Router — mission-critical intent routing (English, CLINC150)

A runnable prototype for **query → intent → fixed canned answer** with a safe
three-tier decision policy: **AUTO** (answer), **CLARIFY** (ask the user), or
**ESCALATE** (human/fallback) — never a silent wrong answer.

Built on a frozen E5 embedding + a logistic intent probe + a fused
out-of-distribution head. The full design rationale, measured results, and the
honest SLA are in **[`ARCHITECTURE.md`](ARCHITECTURE.md)** — read that for *why*.
This README is *how to run it*.

## Pipeline (2 steps, self-contained)

```bash
pip install datasets sentence-transformers scikit-learn numpy torch

cd examples/faq_router

# 1. Build the data from clinc_oos (creates data/: catalog, answer stubs, split records)
python convert_clinc.py --config plus --out data

# 2. Fit the probe + fused OOD head, calibrate the tiers, and evaluate on sealed test.
#    Embeds on-demand and caches under cache/ (regenerable, git-ignored).
#    Pin a GPU with: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
python router_policy.py --target-ood-recall 0.99 --auto-acc-target 0.99
```

`router_policy.py` prints the AUTO / CLARIFY / ESCALATE breakdown, AUTO accuracy
and OOD-leak, and sample dispositions with the tool call the orchestrator would
emit (`deliver_faq_answer` / `ask_clarification` / `escalate_to_human`).

## Files

| File | What it does |
|---|---|
| `convert_clinc.py` | `clinc_oos` → `data/`: `catalog.json` (150 intent ids), `answer_map.json` (**stub answers — author real ones**), split-preserving `records_{train,validation,test}.jsonl`. Identifies OOS **by name**, never by integer id. |
| `embedder.py` | Embedding wrapper with the **correct per-model prompt convention** (e5-instruct vs e5 `query:`/`passage:` vs bge). Wrong convention → meaningless thresholds. |
| `router_policy.py` | The deliverable. E5 embed → intent probe + fused OOD head → **AUTO / CLARIFY / ESCALATE**, with the doppelgänger collision-rule hook (`scope_gate`). Self-contained: embeds on demand, computes centroids/kNN/Mahalanobis inline. |
| `ARCHITECTURE.md` | Full design, measured results, the impossibility proof for 99/99, and the certification checklist. |

## Honest bottom line (details in `ARCHITECTURE.md`)

- **99% OOD-reject AND 99% in-scope auto-coverage is impossible** on hard
  near-OOD (textually-identical doppelgängers). Only clarification / external
  context adds the missing fact.
- Adopt three separate commitments instead: **certified <1% AUTO error**,
  **reported AUTO coverage** (~80% planning on hard OOD), **≥99% end-to-end
  resolution** via AUTO + clarify + human.
- Biggest lever to raise AUTO coverage: **verified hard-negative OOD data** near
  intent boundaries (measured: moves 99%-reject coverage ~9% → ~75%), then
  contrastive encoder finetuning. RAG / bigger judges measured **no gain**.

## Caveats

- `answer_map.json` and intent descriptions are **placeholders** (CLINC has no
  canned answers) — author the real ones before use.
- Thresholds are currently calibrated on validation (exploratory). To *certify*
  the <1% bound you need a nested calibration split + a sealed hard-negative /
  doppelgänger test set (Clopper–Pearson / Learn-then-Test).
- `data/` and `cache/` are regenerable and git-ignored; a fresh clone rebuilds
  them by running the two steps above.
