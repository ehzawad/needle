"""Convert clinc_oos (config 'plus') into router inputs.

Outputs (in --out dir):
  catalog.json      : [{"id": intent, "description": ...}] for the 150 in-scope intents
  answer_map.json   : {intent: "<STUB answer>"} — you MUST author the real answers
  records_{split}.jsonl : {"query", "intent_id"|null, "is_oos"} per official split
  manifest.json     : dataset + label-set fingerprint

Correctness guards (from the council review):
  - OOS is identified BY NAME ("oos"), never by a hard-coded integer id
    (in the pinned dataset it is id 42, and using integers as dense indices
    would silently shift every class after it).
  - Labels are decoded via ClassLabel.int2str; we never index a list by the raw int.
  - Official train/validation/test splits are preserved — needle's stock
    `finetune` re-splits a single JSONL and would contaminate the benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

OOS_NAME = "oos"
EXPECTED_IN_SCOPE = 150


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="plus", help="clinc_oos config (plus/small/imbalanced)")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    from datasets import load_dataset

    try:
        ds = load_dataset("clinc_oos", args.config)
    except Exception:
        ds = load_dataset("clinc_oos", args.config, trust_remote_code=True)

    split0 = next(iter(ds.values()))
    label_col = "intent" if "intent" in split0.features else "label"
    feat = split0.features[label_col]
    names = list(feat.names)
    assert OOS_NAME in names, f"expected an '{OOS_NAME}' class in clinc_oos"

    # All splits must share the identical ClassLabel ordering.
    for name, split in ds.items():
        assert list(split.features[label_col].names) == names, f"split '{name}' label set differs"

    in_scope = [n for n in names if n != OOS_NAME]
    assert len(in_scope) == EXPECTED_IN_SCOPE, f"expected {EXPECTED_IN_SCOPE} in-scope intents, got {len(in_scope)}"
    # CLINC ids are already snake_case; assert the canonicalization is a no-op / injective.
    assert all(n == _canonical(n) for n in in_scope), "some intent name is not canonical snake_case"
    assert len(set(in_scope)) == len(in_scope), "duplicate intent names"

    catalog = [{"id": n, "description": _humanize(n)} for n in in_scope]
    answer_map = {n: f"[TODO: author the canned answer for intent '{n}']" for n in in_scope}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "catalog.json").write_text(json.dumps(catalog, indent=2))
    (out / "answer_map.json").write_text(json.dumps(answer_map, indent=2))

    counts = {}
    for split_name in ("train", "validation", "test"):
        if split_name not in ds:
            continue
        rows = []
        for ex in ds[split_name]:
            intent = feat.int2str(ex[label_col])
            is_oos = intent == OOS_NAME
            rows.append({
                "query": ex["text"],
                "intent_id": None if is_oos else intent,
                "is_oos": is_oos,
            })
        _write_jsonl(out / f"records_{split_name}.jsonl", rows)
        counts[split_name] = len(rows)

    (out / "manifest.json").write_text(json.dumps({
        "dataset": "clinc_oos",
        "config": args.config,
        "num_in_scope": EXPECTED_IN_SCOPE,
        "label_col": label_col,
        "oos_name": OOS_NAME,
        "names_sha1": _sha1(json.dumps(names)),
        "split_counts": counts,
    }, indent=2))

    # Final invariants.
    assert set(answer_map) == {c["id"] for c in catalog}
    assert OOS_NAME not in answer_map
    print(f"wrote catalog.json ({len(catalog)} intents), answer_map.json (STUBS), "
          f"records_*.jsonl {counts}, manifest.json -> {out}/")
    print("NOTE: answer_map.json contains placeholder answers — author the real ones before use.")


def _humanize(n: str) -> str:
    return n.replace("_", " ")


def _canonical(n: str) -> str:
    return n.strip().lower().replace("-", "_").replace(" ", "_")


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def _write_jsonl(path: Path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
