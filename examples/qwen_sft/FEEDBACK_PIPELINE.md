# Robust scope-gate pipeline — conversation signal trains the adapter

> Reconciled with a 4-role Codex Council (clarify-calibration, conversation-signal,
> adapter-training, robust-pipeline). This document is the design of record. The
> earlier domain-SFT plan was retired (it fights the editable-file requirement); only
> a small gate-only "safe-to-answer" head survives as an optional challenger (below).

## The problem this solves

The base file-scope bot ([`scope_bot.py`](scope_bot.py), no training) was **safe but
mis-calibrated**: it over-clarified clearly in-scope queries and under-clarified
genuinely ambiguous ones. The 50-scenario eval ([`eval50.py`](eval50.py)) showed the
two failure directions:

- **Over-clarify** (lost coverage): "raise my daily card spending limit", "how long
  till my card arrives", "put card on google pay", "why charged 3% abroad".
- **Under-clarify** (risky guess): bare "change my PIN" (card vs SIM), "what's my
  balance" (statement vs account — account is out of scope), "a transfer" (own vs
  external).

**Root cause:** the gate clarified on missing *operational* detail (amount, method)
but answered *through* a missing *scope-defining* detail. And — the key finding when
we tried to fix it with a better prompt — **the LLM will not reliably obey a prompt
rule** like "clarify when the scope detail is missing": in a bank context it has a
strong default reading ("PIN = card PIN"). Prompt tuning was whack-a-mole (fixed
two cases, regressed a third). So the fix cannot live only in the prompt.

## The two mechanisms (both training-free, file-edit-safe)

### 1. Deterministic scope policy — the fail-closed floor ([`scope_policy.py`](scope_policy.py))

Collision-prone cards declare machine-checkable **`required_discriminators`** with
literal cues:

```json
"required_discriminators": [{
  "slot": "pin_object",
  "in_scope_cues": ["card pin", "debit", "credit card", "atm"],
  "out_of_scope_cues": ["sim", "voicemail", "app password", "cvv", "otp"],
  "if_unresolved": "CLARIFY",
  "clarifying_question": "Do you mean your Northwind card PIN, or a different PIN?"
}]
```

The LLM only proposes a card. Python then resolves each discriminator against the
query text and **enforces the disposition, downgrade-only** (ANSWER → CLARIFY/ABSTAIN,
never a promotion):

| Query resolves the slot to… | Enforced disposition |
|---|---|
| an out-of-scope cue (`sim`, `chase`, `checking balance`…) | **ABSTAIN** |
| no cue at all (bare "change my PIN") | **CLARIFY** |
| an in-scope cue (`debit`, `checking to savings`…) | ANSWER stays |
| card has no discriminator (`REQUIRES: none`) | ANSWER stays |

Because it can only make a decision *safer*, it **hardens** the 0-leak property: even
if the LLM slips and answers "change my SIM PIN", the `sim` cue forces ABSTAIN. Cards
without a collision print an explicit **`REQUIRES: none`** — without it the model
invents a requirement and over-clarifies a fully-specified query (a measured
regression). Only 6 of 16 cards need a discriminator; the rest use `REQUIRES: none`.

The cost is recall: a paraphrase with no matching cue over-clarifies. That is the safe
direction, and it is recovered by mechanism 2.

### 2. Exemplar bank — recall learned from real conversations ([`feedback_log.py`](feedback_log.py) → [`mine_signals.py`](mine_signals.py) → [`adapter/learn.py`](adapter/learn.py))

This is the user's insight made mechanical: **production chat labels the gate for
free.** Every served turn is logged; the *next* user turn tells us whether the *prior*
gate decision was right:

| Prior gate | Next user turn | Mined label (target for the ORIGINAL query) |
|---|---|---|
| ANSWER | "no, my SIM" (correction) | original was generic → **CLARIFY**; original itself named an OOD thing → **ABSTAIN** (a leak) |
| ANSWER | "perfect, thanks" | **ACCEPT** (card choice validated) |
| ANSWER | *silence / session ends* | **CENSORED — no label** (silence ≠ acceptance) |
| CLARIFY | user just names the slot | **JUSTIFIED — no label** (the clarify worked) |
| CLARIFY | "obviously the card" / original already resolved the slot | **OVER_CLARIFY** → ANSWER |
| ABSTAIN | "I mean the foreign-transaction fee" (rephrase in-scope) | **OVER_ABSTAIN** → review |

Reviewed high-confidence labels become a small, per-card-capped **exemplar bank**
injected into the gate prompt (opt-in; off in the base-gate eval). An earlier held-out
probe showed 5 real-conversation exemplars flip *paraphrases* of the logged queries
from wrong-guess to correct-clarify with no regression — i.e. the signal generalizes,
without retraining.

**Guardrails baked into the miner (all Council findings):** silence is not acceptance;
a terse valid clarification answer is not over-clarification; OOD vocabulary is derived
from the *cards* (not a hardcoded regex) so it tracks edits; every ANSWER-expanding
label is forced to human review and can never auto-enter the bank; every label is
pinned to the KB hash and quarantined if the card file changed.

## End-to-end pipeline

```
human-approved cards ─▶ INGEST + hash snapshot            [training-free + human authority]
                       RETRIEVE (all 16 now; hybrid at 150–500)
                       LLM GATE proposes card/disposition [frozen Qwen3-4B]
                       DETERMINISTIC SCOPE POLICY (fail-closed, downgrade-only)
                         ├─ ABSTAIN / CLARIFY / ANSWER
                       GROUND generation from key_facts only
                       (post-gen verifier — needed before production factuality claims)
                       SERVE ─▶ LOG turn (+ next-turn link)
                                 └▶ MINE weak labels ─▶ HUMAN REVIEW ─▶ EXEMPLAR BANK
                                      └▶ (escalate to a trained gate-only head only if the
                                          bank plateaus on a sealed held-out set)
                       Offline regression ─▶ shadow ─▶ canary ─▶ promote; rollback on any leak
```

## What is trained vs not

Everything shipping here is **training-free**: frozen Qwen3-4B (gate + generator),
deterministic policy, prompt/exemplar updates. The only *possible* trained component
is a small **gate-only "safe-to-answer" head** over the frozen hidden state — a
**veto** that can turn a proposed ANSWER into CLARIFY/ABSTAIN, never the reverse. Per
the Council it is the escalation **only if** a family-disjoint held-out set of real,
reviewed conversations shows the exemplar bank cannot meet: 0 leaks, ≥85% coverage,
≤10% over-clarify, ≤5% under-clarify — and only with ≥10k adjudicated families. Domain
SFT / training the generator stays rejected (it fights the editable-file requirement).

## Safety & promotion (the 0-leak property is non-negotiable)

- **Offline gate:** 0/25 leaks on the frozen safety set + every logged production-leak
  regression; for this calibration release, all 20 in-scope ANSWER and all 5 ambiguous
  CLARIFY (so a regression can't hide behind a net coverage gain).
- **Statistical honesty:** 0/25 has a one-sided 95% upper bound ≈ 11%. A `<1%` claim
  needs 0 failures across ≥299 labeled cases on a named distribution (Clopper–Pearson).
  The 50 scenarios are a **dev set, not certification.**
- **Canary circuit breaker:** zero tolerance — one confirmed OOD answer (or a canary
  probe answered) rolls back the *exemplar-bank/head version*, not the card file.
- **Card edits:** every exemplar/head artifact binds to the KB hash; on mismatch,
  exemplar injection disables and the base file-scope gate serves. Stale labels are
  quarantined, not silently reused.

## Metrics

| Metric | Definition | Baseline | Target |
|---|---|---:|---:|
| Harmful-leak rate | OOD/adversarial answered ÷ OOD/adversarial | 0/25 | 0 observed; 95% upper <1% to certify |
| In-scope coverage | right-card ANSWER ÷ in-scope | 80% | 85–95%; dev-gate 20/20 |
| Over-clarify rate | in-scope CLARIFY ÷ in-scope | 20% | 0–10% |
| Under-clarify rate | ambiguous ANSWER ÷ ambiguous | 60% | 0–5% |
| Wrong-card rate | wrong-card ANSWER ÷ in-scope | — | 0 |

## Demo (16 cards) vs the 150–500 target

Genuinely needed now: executable discriminators + deterministic policy, KB hashing,
immutable per-turn logs, conservative card-derived mining, human review, KB-versioned
exemplar bank, one post-gen verifier. **Over-engineering now, needed at 500:** a
retrieval index + confusable-sibling expansion, a trained gate head, a canary
controller, a review UI, durable governed logging. **Wrong at any scale:** domain
SFT, training the generator, a dual-verifier ensemble, the LLM's self-reported
confidence as a certificate.

## Files

`scope_policy.py` (deterministic discriminator enforcement, no torch) ·
`scope_bot.py` (gate + policy + grounded gen + opt-in exemplars) ·
`seed16/cards.json` (scope file; 6 cards carry executable discriminators) ·
`feedback_log.py` (per-turn telemetry) · `mine_signals.py` (weak-label miner) ·
`adapter/learn.py` (reviewed labels → exemplar bank) · `eval50.py` (dev eval, right-card
+ 5 metrics) · `logs/conversations.sample.jsonl` (illustrative traffic).
Runtime artifacts (`logs/*.jsonl` real logs, `exemplars.json`) are git-ignored.
