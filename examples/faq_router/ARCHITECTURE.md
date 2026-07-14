# FAQ Router — Final Architecture (mission-critical, English, server-side)

Consolidated conclusion of four Codex Council reviews + measured experiments on
CLINC150 (150 intents + out-of-scope). All numbers measured live on an RTX A5000.

## The goal, and the honest verdict

**Goal:** auto-answer ~99% accurately, block "semantic doppelgängers" (hard
near-OOD), still let genuinely in-scope queries through.

**Verdict (proven, not opinion):** you cannot have **99% OOD rejection AND ~99%
in-scope auto-answer coverage simultaneously** on hard near-OOD. When an
in-scope and an out-of-scope query are *textually identical* ("change my PIN" =
card-PIN supported vs SIM-PIN not), the distinguishing fact is absent from the
text. No embedding, RAG layer, cross-encoder, LLM judge, ellipsoid, or ensemble
can recover an absent referent. Only **clarification** (ask the user) adds the
missing fact. So the goal is reached by *decomposition*, not by a better classifier.

## The architecture (built and measured — `router_policy.py`)

```
query
  → e5-large-instruct embedding (frozen, precomputed KB)
  → intent probe (97.4%)  +  fused OOD head  [cosine · Mahalanobis · relMahalanobis · kNN · confidence]
  → THREE-TIER decision (never a silent rejection):

     ESCALATE   fused OOD prob ≥ τ_ood        → likely doppelgänger/OOD → human/fallback
     CLARIFY    in-scope, but margin < τ_margin (or a required slot missing)
                                              → ask "did you mean A or B?" using the
                                                candidate intents' CANONICAL questions
     AUTO       in-scope AND clear winner AND hard-gates pass → the fixed answer

  → orchestrator emits a TOOL CALL:  answer{intent} | ask_clarification{options} | escalate_to_human
```

Two thresholds, both calibrated on validation:
- `τ_ood` — escalate boundary (set from your target OOD rejection / wrong-answer cost).
- `τ_margin` — auto-vs-clarify boundary (set so the AUTO region hits a target accuracy).

### Two rules council-4 added (do not skip these)

1. **Doppelgänger collision rule (registry, overrides confidence).** An *exact*
   doppelgänger has HIGH confidence and pure neighbors, so the OOD score will NOT
   catch it — it reaches AUTO. You must exclude known-collision families as a
   class via the intent registry, using a required scope-defining slot:
   ```
   intent: card_pin_change    scope-slot: pin_object
     pin_object = bank_card  -> AUTO-eligible
     pin_object = SIM/voicemail -> ESCALATE (out of scope)
     pin_object = absent     -> CLARIFY ("card, SIM, or voicemail PIN?")
   ```
   This is the `scope_gate` hook in `router_policy.py` — it runs BEFORE the
   confidence logic. Known collisions are policy knowledge, not something a
   threshold can discover.
2. **CLARIFY is support-gated — uncertainty alone must never trigger it.** First
   check in-scope SUPPORT (fused OOD score, cosine floor, kNN distance/purity).
   LOW support → ESCALATE (likely OOD). Only a well-supported query with a
   *bounded, discriminative* ambiguity (2–3 close siblings with a real candidate
   menu) or a missing scope slot may CLARIFY. Showing the nearest two intents to
   arbitrary OOD traffic would coerce a doppelgänger into a wrong answer.
   The kNN distinction is decisive: close neighbors split across 2–3 siblings =
   ambiguity (CLARIFY); distant/diffuse neighbors = weak support (ESCALATE).

## Measured operating points (sealed CLINC test: 4500 in-scope + 1000 OOD)

| Setting | AUTO (traffic) | AUTO in-scope acc | AUTO OOD-leak | legit in-scope: AUTO / CLARIFY / ESCALATE |
|---|---:|---:|---:|---|
| 90% OOD-reject | 82.0% | 98.4% | 2.90% | 97.3% / 1.6% / 1.1% |
| **99% OOD-reject** (mission-critical) | 78.3% | 98.4% | **1.07%** | 94.7% / 0.6% / 4.7% |

Reading: to block doppelgängers harder (99% reject), AUTO coverage drops from 82%
→ 78% and ~5% of legitimate queries get escalated (vs 1%). Even at the 99% target,
**~1% of OOD still leaks into AUTO on CLINC's *easy* OOD** — on real hard near-OOD
it is worse (hard-negative OOS collapses AUROC 0.968 → 0.914). That residual is the
irreducible doppelgänger band.

## Why Cactus/Needle is NOT on the routing path

Cactus runs Needle (a generative tool-caller). For fixed-answer classification the
E5 probe + fused OOD head beats generative routing, and OOD detection cannot be
done reliably by a generative model or a prompt. So the **orchestrator** (the thin
service around the E5 router) owns the auto/clarify/escalate decision and emits the
tool call. Needle/Cactus earns its place only when you later need on-device
generative tool-calling with real arguments.

## The honest target — THREE separate commitments (council-4)

Replace "99% rejection + 99% auto coverage" (impossible) with three audited numbers:

1. **End-to-end resolution SLA ≥ 99%** — of legitimate queries, correctly resolved
   via AUTO, one clarification, or human handling within a defined service window.
2. **Certified AUTO error < 1%** — one-sided 95% Clopper–Pearson upper bound on
   wrong AUTO answers (OOD false-accepts + in-scope misroutes share the SAME 1%
   budget), on a sealed set with verified hard negatives.
3. **Reported AUTO coverage** at that safety level — planning ~80% on hard OOD.
   Also report hard-OOD false-acceptance separately (target ≤1%).

The recovery path is first-class, not a fallback: with AUTO coverage 80% and error
`e_A`, end-to-end resolution `R = 0.8(1−e_A) + 0.2·r_D`, so reaching 99% needs
deferred-path correctness `r_D ≥ 0.95 + 4·e_A` (e.g. e_A=0.5% → r_D≥97%).

The product claim to make: *"We certify <1% error on automatic answers, report the
automatic coverage at that safety level (~80% under hard OOD), and recover deferred
legitimate queries via clarification or human handling to meet a 99% end-to-end SLA."*

**Certification:** nested calibration split (current code fits fusion + threshold on
the same val split — not certifiable); Learn-then-Test / conformal risk control on a
**sealed hard-negative test set**; ~300–600 zero-error accepted cases for a 95% <1%
bound; certify initial-query AUTO and post-clarification AUTO separately; report
worst-slice bounds for doppelgänger families and safety classes.

**Runtime ownership:** the E5 scoring service returns *features*; the **orchestrator**
applies the versioned policy and emits one of `deliver_faq_answer` / `ask_clarification`
/ `escalate_to_human` as a tool call (schema in council-4's report — includes a
mandatory `neither` option). Cactus can dispatch that tool call; Needle stays off the
routing path. Escalations create a durable case BEFORE acknowledging success; a
timeout never falls back to a Needle guess.

## Highest-leverage next steps (in order)

1. **Collect + human-verify hard-negative OOD** near your intent boundaries
   (~5–10 per boundary → 750–1500 for 150 intents). This moves coverage-at-99%-
   rejection from ~9% (no hard negatives) to ~75% (hard+general) — the single
   biggest lever. Data moves the wall; architecture does not.
2. **Nested calibration + sealed test** (current code fits fusion and threshold on
   the same val split — fine for exploration, not certification).
3. Optionally **contrastive encoder finetuning** (ellipsoid/ADB line): +5–10 pts
   coverage at fixed rejection, only after (1).
4. RAG answerability / bigger judges: measured **no gain** over the fused head
   off-the-shelf; only a *product-fine-tuned* verifier might add 1–3 pts. Not first.

## What was tried and rejected (measured)

- `oos` as a 151st class: OOD recall only 49% (geometrically wrong).
- RAG cross-encoder answerability (ms-marco/qnli/quora): ≤ +0.006 AUROC, no gain vs fused 0.988.
- One-canonical-question-per-tag as the router: R@1 87.6% vs centroid 95.4% (worse) —
  but it IS the right content for the CLARIFY menu.
- Needle generative routing for this fixed-answer task: unnecessary; probe wins.

## File map

`convert_clinc.py` (clinc_oos → data/) · `embedder.py` (E5 prompt conventions) ·
**`router_policy.py` (the three-tier selective router — the deliverable; fits the
probe + fused OOD head, calibrates the tiers, emits the tool call)** · this doc.

The exploratory scripts that produced the measured numbers above (probe/OOD
strategy sweeps, Mahalanobis/kNN comparison, RAG-answerability test, medoid test)
were removed after their findings were folded into this document; the results
they produced are reported inline here.
