# File-defined-scope RAG bot (the chosen architecture)

A conversational bot whose **domain is a file** (scope cards). Every query runs a
**training-free LLM scope-gate** (ANSWER / CLARIFY / ABSTAIN) over the query + the
cards, then grounded generation from the selected card's approved facts. Edit the
file → the domain changes, no retraining. Reconciled by a 3-role Codex Council
that ruled **against** domain SFT.

## Why not SFT (council verdict)

Baking domain facts/scope into weights fights the "domain is an editable file"
requirement: a KB edit leaves stale facts in the adapter, the model can answer
from memorized facts even when the file changed, and every scope change needs
retrain + recalibrate + recertify — and it still doesn't break the doppelgänger
wall. So: **no domain SFT.** The domain-SFT scaffold was removed as a rejected
experiment; only a possible small *gate-only* head remains as an optional
*challenger* (see `FEEDBACK_PIPELINE.md`) — never for domain knowledge.

## Architecture (`scope_bot.py`)

```
versioned scope file (cards: goal, included, excluded, key_facts, capability)
        ↓   (16 cards fit the judge context; add frozen retrieval at 150-500 — see below)
per-query SCOPE GATE = training-free structured LLM judge over query + cards
        →  ANSWER | CLARIFY | ABSTAIN     (runs BEFORE generation)
        ↓ ANSWER
grounded generation using ONLY the selected card's approved facts
        ↓
(optional) HHEM + bidirectional NLI veto  →  answer | canonical fallback | abstain
```

The gate must run **before** generation — a post-gen groundedness check alone
would happily serve a grounded-but-scope-wrong answer (SIM-PIN query → card-PIN
card → fluent, grounded, wrong). Retrieval selects candidate cards; it never
authorizes the answer (a doppelgänger retrieves the wrong card with high
similarity). Uses local Qwen3-4B-Instruct-2507 (4-bit) as judge + generator.

## Measured demo (A5000, 16-card KB, no training)

Correct: in-scope → grounded ANSWER; "change PIN on my **SIM card**" → ABSTAIN
(caught via the file's exclusion — the win the E5 router couldn't get);
"transfer to a friend at **Chase**" → ABSTAIN; "banana bread" → ABSTAIN.
Honest gaps, exactly as predicted: "How do I change my PIN?" (no object) is
answered as card-PIN instead of CLARIFY, and one clearly-in-scope query
over-clarified.

## The doppelgänger ceiling (unchanged, approach-independent)

- **Explicit** doppelgänger ("SIM PIN") → catchable, IF the card encodes the
  exclusion and the judge applies it over topical similarity. ✓ demonstrated.
- **Identical text** ("change my PIN", meaning card vs SIM) → no text-only gate
  can separate; the safe move is CLARIFY. Planning ceiling ~35–60% hard-OOD catch
  at ≤5% false-abstention; ~5% on truly identical text. Clarification/context is
  the only way to break it.

## Update — the calibration fix shipped (see [`FEEDBACK_PIPELINE.md`](FEEDBACK_PIPELINE.md))

`required_discriminators` is now implemented, but **not as a prompt hint** — the
Council showed the LLM won't reliably obey a prompt rule. It is enforced by a
**deterministic, fail-closed policy** ([`scope_policy.py`](scope_policy.py)) that reads
machine-checkable cues on the card and can only downgrade a decision (ANSWER →
CLARIFY/ABSTAIN), which also *hardens* the 0-leak property. Cards without a collision
print `REQUIRES: none` (without it the model invents a requirement and over-clarifies).
On the 50-scenario dev set this took the gate to **0/25 leaks, 20/20 right-card
in-scope coverage, 5/5 ambiguous→clarify** — with no exemplar bank. Real-conversation
signal then trains an **exemplar bank** for recall on phrasings the static cues miss
(the mining loop: [`feedback_log.py`](feedback_log.py) → [`mine_signals.py`](mine_signals.py)
→ [`adapter/learn.py`](adapter/learn.py)). Full design in `FEEDBACK_PIPELINE.md`.

## Next steps (in priority order)

1. ✅ **Cards enriched with executable `required_discriminators`** + deterministic
   policy → "How do I change my PIN?" → CLARIFY; "…SIM PIN" → ABSTAIN; done.
2. **Add frozen hybrid retrieval** (BM25 + a small dense encoder) to shortlist
   3–6 cards + confusable siblings, for the 150–500-intent scale (16 fit today).
3. **Post-gen HHEM/NLI veto** — reuse the `faithful-answer` branch's verifier,
   adapted from canonical-rewrite to fact-ID grounding (fail-closed).
4. **Multi-turn resolver** — scope-check the *resolved standalone* query; never
   inherit a prior ANSWER; don't let the resolver fill a missing discriminator.
5. **The decisive experiment + certification**: family-disjoint, human-reviewed
   sealed test; compare prompt-only vs RAG+gate vs the SFT challenger at matched
   ≤5% false-abstention; report wrong-among-answered with a Clopper–Pearson bound
   (≥299 accepted for <1%). Ship RAG+gate unless SFT gains ≥10 abs. points hard-OOD.

## Honest guarantee

For a frozen KB version + model + prompts + decoding + threshold on a named
distribution: wrong-among-answered ≤ U at 95% (Clopper–Pearson), at reported
coverage; abstain/clarify outside the certified region. Not a promise for
arbitrary/adversarial input, and it does not separate identical-text doppelgängers.

## Files

See the top-level [`README.md`](README.md) for the current file map. The earlier
domain-SFT scaffold was removed as a rejected experiment; only a gate-only head
survives as an optional challenger, documented in [`FEEDBACK_PIPELINE.md`](FEEDBACK_PIPELINE.md).
