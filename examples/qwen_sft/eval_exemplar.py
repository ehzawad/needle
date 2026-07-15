"""Fair check of the exemplar bank: does learning from real conversations help on
NEW phrasings, not just the exact logged queries?

We hold the exemplar'd queries OUT of the test and instead probe PARAPHRASES of
them, running each through the SAME model twice — base gate vs. gate + exemplar
bank — so any movement is generalization, not teaching-to-the-test. eval50.py
stays the clean base-gate headline; this is the separate learning-signal probe.
"""
from scope_bot import ScopeBot

# Held-out paraphrases of queries the bank learned from (bank has the ORIGINAL
# wording; these are reworded), plus a control that was already correct.
PROBES = [
    ("i need to make a transfer, can you help me out?", "CLARIFY", "transfer: own vs external unresolved"),
    ("can you help me set a new pin?", "CLARIFY", "pin: card vs SIM unresolved"),
    ("what's the charge for paying in a foreign currency on my card?", "ANSWER:international_fees", "FX fee is a static in-scope fact"),
    ("how do I report my card stolen?", "ANSWER:report_lost_card", "control — should stay ANSWER either way"),
]


def score(bot, tag):
    print(f"\n===== {tag} =====")
    for q, ideal, note in PROBES:
        g = bot.gate(q)
        got = g["disposition"] + (f":{g['card_id']}" if g["disposition"] == "ANSWER" else "")
        ok = "OK " if got == ideal or (ideal.startswith("CLARIFY") and got == "CLARIFY") else "-- "
        print(f"  {ok} ideal={ideal:26} got={got:26} | {q}")
        print(f"        reason: {g['reason'][:80]}")


def main():
    bot = ScopeBot(use_exemplars=True)  # loads bank + model once
    bank = list(bot.exemplars)
    print(f"loaded exemplar bank: {len(bank)} entries")
    bot.exemplars = []          # condition A: base gate (discriminators only)
    score(bot, "BASE gate (no exemplar bank)")
    bot.exemplars = bank        # condition B: + real-conversation exemplars
    score(bot, "GATE + exemplar bank")


if __name__ == "__main__":
    main()
