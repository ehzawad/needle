# Qwen3-4B refusal-aware SFT — generative bot that abstains out-of-scope

Jettisons the E5 router. A single **generative** model (SFT'd Qwen3-4B) answers
in-scope questions by rephrasing approved facts, and says *"I don't know / out of
scope"* on anything outside its SFT domain — including semantic doppelgängers.
Objective: **never a wrong answer** = *answer when confident, abstain when not*.
Reconciled with a 4-role Codex Council; VRAM-measured on this A5000.

## Architecture

```
query
  │
  ▼
[gate]  safe-to-answer head over Qwen's final prompt hidden state  → P(in-scope AND answerable-correctly)
  │        (a tiny supervised probe — the production boundary, drawn in the model's OWN semantics; no E5)
  ├─ P < τ  →  ABSTAIN (canonical refusal)
  └─ P ≥ τ  →  generate with the SFT'd model
                 └─ if generation itself refuses / malformed → ABSTAIN
```

Two learned pieces on top of Qwen3-4B, both cheap:

1. **Refusal-aware SFT (R-Tuning-style), QLoRA.** Train on a mix of
   `{in-scope question → generative answer}` and `{OOD question → refusal}`. This
   teaches the answer/abstain *behavior* and catches **far**-OOD well. It does
   **not**, by itself, reliably separate hard near-OOD (that's the proven wall).
2. **Safe-to-answer gate head** over the post-SFT hidden state at the last prompt
   token, supervised to predict *"in-scope AND the model's answer is correct"* —
   not merely "one of the intents." This is the production abstention decision.
   It's a supervised cousin of **Semantic Entropy Probes** (uncertainty decoded
   from hidden states) — chosen over full semantic entropy (5–10× generation cost,
   offline-only), a generic refusal-direction probe (measures the *act* of
   refusing, not scope), raw logprob (uncalibrated), and a separate embedder
   (reintroduces the E5 doppelgänger failure mode). Calibrate with temperature/Platt.

**Why this draws the boundary the E5 classifier couldn't:** the head reads Qwen's
contextual representation *and* is supervised on actual answer-correctness, so it
abstains on in-scope questions the model would get wrong, not just on far topics.
Full **semantic entropy** (sample k, NLI-cluster by meaning, high entropy → guess)
is used **offline** to *label* which in-scope cases are unreliable, then distilled
into the head. Correctness first; the head is the fast single-pass deploy form.

## The honest ceiling (unchanged, proven)

If an in-scope query and an OOD query are identical text and differ only by an
absent fact, no text-only mechanism can separate them: at a 5% in-scope
false-abstention budget, ≲5% of *truly indistinguishable* doppelgängers are
caught. Planning target: **97–99.5% far-OOD** abstention, **~35–60% hard-OOD**
at ≤5% false-abstention. "Never wrong" = a conservative selective predictor with
a certified bound, not perfect separation.

## Data recipe (first 150-intent run)

Intent labels aren't truth — author a human-approved **scope card** per intent
first (supported goal, included/excluded entities, capability class, required
facts, confusable siblings, boundary atoms). Then:

| Split | in-scope | FAR-OOD | HARD near-OOD | total |
|---|---:|---:|---:|---:|
| train | 15,000 (100/intent) | 3,000 | 4,500 | 22,500 |
| val | 3,000 | 600 | 900 | 4,500 |
| sealed test | 4,500 | 900 | 1,350 | 6,750 |

- Answer:refusal ≈ **2:1**; within refusals FAR:HARD ≈ **40:60**.
- **20 answer realizations/intent** (structural variety, same facts — generative,
  not one canned string). Grade by fact-entailment, never BLEU/exact-match.
- **Hard negative** = start from an in-scope question, change *exactly one*
  scope-defining atom (card-PIN → SIM-PIN); reject if it maps to another intent or
  merely drops the atom (that's ambiguity, not OOD). ~34% of raw candidates survive
  review → ~22,000 raw candidates for 6,750 accepted.
- **Pair every hard-negative with a matched in-scope twin** to prevent
  over-refusal ("anything mentioning PIN → refuse").
- Leakage-safe: paraphrase/mutation **families never cross splits**; keep CLINC's
  official OOS test as an untouched external benchmark.

**Generation pipeline** (the ultracode workflow): scope-card agents → per-intent
answer/question/hard-neg/FAR agents → 4 verifier roles (global-scope, adversarial,
mutation, answer) → human review (2 reviews per hard-negative). A **7-intent seed**
(`seed/`, "Northwind Bank" fictional policy) was produced by `qwen-sft-seed-data`;
its generation succeeded but the *verifier* phase was interrupted by a session
limit, so `assemble_seed.py` recovered it from the workflow journal and did an
inline review — full human review is still owed. `card_declined` generation hit
the same limit and is absent.

## QLoRA training config (measured on THIS A5000)

`train_qlora.py` — Qwen3-4B-Instruct-2507, NF4 double-quant, bf16 compute, LoRA
r=16/α=32 on all 7 projections (~33M trainable, 0.82%), max_len 1024, microbatch 1
× grad-accum 16, gradient checkpointing, FlashAttention-2, packing (bfd),
paged-adamw-8bit, lr 1e-4 cosine, 3 epochs. **Measured peak 14.93 GiB alloc /
15.98 GiB reserved → fits 24 GB with ~7.5 GiB headroom** (18 GiB smoke gate).

**Resumable:** checkpoint every 25 updates; a `_SUCCESS` marker is written only
after a full save; SIGTERM/SIGINT checkpoint-then-stop; resume picks the latest
*validated* checkpoint (optimizer + scheduler + RNG restored). Smoke-test 1 step →
resume → step 2 before the full run. Env pitfalls handled in `setup_env.sh`
(TRL≥0.23.1 or it trains on the prompt; torch 2.6 or resume breaks; select A5000
by UUID — bare `CUDA_VISIBLE_DEVICES=0` picks the A6000 here).

## Evaluation & the guarantee

Selective risk + coverage, not accuracy. Harmful = wrong in-scope answer OR *any*
substantive answer to OOD. Report FAR and HARD **separately**, a risk–coverage
curve, macro + worst-intent, and a held-out **doppelgänger challenge** (2 matched
pairs/intent). Honest guarantee: *for a frozen model+prompt+decoding+threshold on
a named distribution, wrong-answer-rate-among-answered ≤ U at 95% (Clopper–Pearson;
zero errors in 299 answered → <1%, 598 → <0.5%), coverage = c.* Not a promise for
arbitrary/adversarial inputs. Go/no-go: ε=1% bound, FAR & HARD leak ≤1%, ≥80%
macro in-scope coverage, zero OOD answers on the doppelgänger set.

## Files

`SFT_ABSTAIN.md` (this) · `setup_env.sh` (pinned venv + A5000 UUID) ·
`train_qlora.py` (resumable QLoRA trainer, VRAM-guarded) · `assemble_seed.py`
(recover workflow output → JSONL) · `seed/` (7-intent seed: `cards.json`,
`train.jsonl`/`validation.jsonl` = prompt/completion, `seed_annotated.jsonl` +
`manifest.json`) · *(next)* `train_gate.py` (safe-to-answer head) · `evaluate.py`
(selective-risk + doppelgänger eval). Full training data goes to `data/sft/`
(git-ignored). Smoke-test on the seed with
`TRAIN_FILE=seed/train.jsonl VAL_FILE=seed/validation.jsonl SMOKE=1 python train_qlora.py`.

## Status / next steps

1. ✅ config + resumable trainer + pinned env.
2. ✅ 7-intent seed via the `qwen-sft-seed-data` ultracode workflow (`seed/`);
   verifier phase interrupted by a session limit → inline-reviewed, full human
   review pending; `card_declined` to be regenerated.
3. author real scope cards + scale the dataset to 150 intents (agents + review).
4. `bash setup_env.sh`, smoke-test → resume-test → full QLoRA run on the A5000.
5. train the safe-to-answer gate head; calibrate τ; run the doppelgänger eval.
