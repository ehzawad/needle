"""Deterministic scope-policy enforcement — the fail-closed layer the council
requires OUTSIDE the LLM.

The 50-scenario eval proved that a smarter gate PROMPT is not enough: the base
model, told to "CLARIFY when a scope-defining detail is missing", still guessed on
bare "change my PIN" / "a transfer" because in a bank context it has a strong
default reading. Prompt compliance is a second failure mode. So we don't trust the
LLM to apply the discriminator rule — we make it machine-checkable:

  - each collision-prone card declares `required_discriminators` with literal
    `in_scope_cues` / `out_of_scope_cues`;
  - the LLM only proposes a card;
  - THIS module resolves each discriminator against the query text and enforces the
    disposition, and it only ever makes the decision SAFER (ANSWER -> CLARIFY/ABSTAIN),
    never riskier. It never promotes CLARIFY/ABSTAIN to ANSWER.

Because it can only downgrade, it also HARDENS the measured 0-leak property: even if
the LLM slips and answers "change my SIM PIN", the `sim` out-of-scope cue forces
ABSTAIN. Recall lost to missing cues (a paraphrase with no cue -> CLARIFY) is the
safe direction, and is recovered over time by the exemplar bank learned from real
conversations (see feedback_log.py / mine_signals.py / adapter/learn.py).
"""
from __future__ import annotations


def resolve_discriminator(query: str, disc: dict) -> tuple[str, str | None]:
    """Resolve one discriminator against the query. Out-of-scope cues win over
    in-scope cues (fail-closed). Returns (status, matched_cue)."""
    q = " " + query.lower() + " "
    for cue in disc.get("out_of_scope_cues", []):
        if cue.lower() in q:
            return "OUT_OF_SCOPE", cue
    for cue in disc.get("in_scope_cues", []):
        if cue.lower() in q:
            return "IN_SCOPE", cue
    return "UNRESOLVED", None


def _abstain(reason: str) -> dict:
    return {"disposition": "ABSTAIN", "card_id": None, "candidates": [],
            "reason": reason, "policy_action": "abstain"}


def enforce_policy(query: str, gate: dict, by_id: dict) -> dict:
    """Apply the deterministic scope policy to the LLM's proposed decision.
    DOWNGRADE-ONLY: an ANSWER can become CLARIFY or ABSTAIN; a CLARIFY/ABSTAIN is
    left as the LLM decided (we never make a decision riskier than the model's)."""
    disp = gate.get("disposition")
    if disp != "ANSWER":
        return gate  # fail-closed: never promote CLARIFY/ABSTAIN to ANSWER
    card = by_id.get(gate.get("card_id"))
    if not card:
        return {**_abstain("policy: ANSWER named an unknown card"),
                "policy_action": "downgrade_answer_to_abstain"}
    for disc in card.get("required_discriminators", []):
        status, cue = resolve_discriminator(query, disc)
        if status == "OUT_OF_SCOPE":
            return {"disposition": "ABSTAIN", "card_id": None, "candidates": [],
                    "reason": f"policy: {disc['slot']} resolves out-of-scope via '{cue}'",
                    "policy_action": "downgrade_answer_to_abstain"}
        if status == "UNRESOLVED":
            return {"disposition": "CLARIFY", "card_id": None,
                    "candidates": [gate.get("card_id")],
                    "reason": f"policy: {disc['slot']} unresolved (scope-defining detail missing)",
                    "clarifying_question": disc.get("clarifying_question", ""),
                    "policy_action": "downgrade_answer_to_clarify"}
    return gate  # all discriminators resolved in-scope (or none) -> keep ANSWER


def all_out_of_scope_cues(cards: list[dict]) -> dict[str, list[str]]:
    """Map card_id -> its declared out-of-scope cues, so downstream signal mining
    derives OOD vocabulary from the CARDS (versioned) rather than a hard-coded regex."""
    out: dict[str, list[str]] = {}
    for c in cards:
        cues = []
        for d in c.get("required_discriminators", []):
            cues += d.get("out_of_scope_cues", [])
        if cues:
            out[c["intent_id"]] = cues
    return out


def query_hits_ood_cue(query: str, cards: list[dict]) -> str | None:
    """Return the first out-of-scope cue (from ANY card) present in the query, else None.
    Card-derived replacement for a hand-maintained OOD keyword list."""
    q = " " + query.lower() + " "
    for c in cards:
        for d in c.get("required_discriminators", []):
            for cue in d.get("out_of_scope_cues", []):
                if cue.lower() in q:
                    return cue
    return None
