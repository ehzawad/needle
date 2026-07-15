"""Content-addressed source fingerprinting — the supply-chain provenance layer.

Real-world analog: DVC / lakeFS content-addressed data versioning combined with
in-toto / Sigstore source attestation. Just as in-toto pins the exact source that
produced an artifact and a lockfile pins dependency integrity by hash, this module
binds a candidate to the EXACT bytes of the frozen scope sources and the EXACT
50-case eval suite. If either drifts, ingestion must fail.

Two deliberate constraints make this the control-plane-safe fingerprinter:

  * It never imports ``scope_bot`` or ``eval50`` (which would drag in torch and the
    A6000-grabbing model load). It reads the prompt constants and the ``SCEN`` suite
    by parsing the source with :mod:`ast` — a pure, side-effect-free read.
  * It provides BOTH a whole-file digest (:func:`sha256_file`, used for the frozen
    freeze check that ``config.frozen.eval_suite_sha256`` equals the bytes of
    ``eval50.py``) AND a canonical, structure-only digest of the parsed cases
    (:func:`eval_suite_sha256`, used to run/label the suite in the candidate
    manifest's evaluation contract). They intentionally differ: one pins the file,
    the other pins the semantic suite.
"""
from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pipeline.contracts import EvalCase

# NUL byte joins the two system prompts into the single combined fingerprint. This
# matches the reconciled council value 04290adb...5ad928 for scope_bot.py and keeps
# the two prompts unambiguously separable (no prompt text can contain a raw NUL).
_PROMPT_JOIN = "\x00"

_EVAL_CATEGORIES = frozenset(
    {"in_scope", "hard_ood", "far_ood", "ambiguous", "adversarial"}
)


@dataclass(frozen=True)
class SourceFingerprint:
    """The runtime-identity constants extracted from ``scope_bot.py`` by AST.

    ``judge_prompt_sha256`` / ``generation_prompt_sha256`` populate the candidate
    manifest's ``runtime`` block; ``combined_prompt_sha256`` is the single
    gate+generation prompt digest (JUDGE_SYS + NUL + GEN_SYS).
    """

    judge_sys: str
    gen_sys: str
    model_id: str
    revision: str
    judge_prompt_sha256: str
    generation_prompt_sha256: str
    combined_prompt_sha256: str


def canonical_json(value: object) -> bytes:
    """Deterministic, sorted-key, UTF-8 JSON bytes.

    Content addressing is only stable if the serialization is stable, so keys are
    sorted, separators are tight, and non-ASCII is preserved (``ensure_ascii=False``)
    rather than escaped — the same object always yields the same bytes.
    """
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    """Lowercase hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Lowercase hex SHA-256 of a file's bytes, streamed so large files stay cheap."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_assignments(path: Path) -> dict[str, ast.expr]:
    """Map every module-level ``NAME = <expr>`` to its value node, without executing
    the module. Later assignments win (module constants are assigned once)."""
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    out: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value
    return out


def _literal_str(assigns: dict[str, ast.expr], name: str, path: Path) -> str:
    """Safely evaluate a module-level string constant (handles implicit string
    concatenation, which the parser folds into a single literal)."""
    if name not in assigns:
        raise ValueError(f"source_fingerprint: {path} defines no top-level {name!r}")
    try:
        value = ast.literal_eval(assigns[name])
    except (ValueError, SyntaxError) as exc:  # non-literal RHS
        raise ValueError(
            f"source_fingerprint: {name} in {path} is not a literal ({exc})"
        ) from exc
    if not isinstance(value, str):
        raise ValueError(
            f"source_fingerprint: {name} in {path} is {type(value).__name__}, not str"
        )
    return value


def read_scope_constants(path: Path) -> SourceFingerprint:
    """Extract JUDGE_SYS / GEN_SYS / MODEL_ID / REV from a scope_bot-shaped source by
    AST — never importing it (so no torch, no model load, no A6000 grab)."""
    assigns = _module_assignments(path)
    judge = _literal_str(assigns, "JUDGE_SYS", path)
    gen = _literal_str(assigns, "GEN_SYS", path)
    model_id = _literal_str(assigns, "MODEL_ID", path)
    revision = _literal_str(assigns, "REV", path)
    return SourceFingerprint(
        judge_sys=judge,
        gen_sys=gen,
        model_id=model_id,
        revision=revision,
        judge_prompt_sha256=sha256_bytes(judge.encode("utf-8")),
        generation_prompt_sha256=sha256_bytes(gen.encode("utf-8")),
        combined_prompt_sha256=sha256_bytes(
            (judge + _PROMPT_JOIN + gen).encode("utf-8")
        ),
    )


def load_eval50_cases(path: Path) -> tuple[EvalCase, ...]:
    """Parse the ``SCEN`` list literal in an eval50-shaped source into typed cases.

    ``SCEN`` rows are variable-length tuples: in-scope rows carry an expected card as
    a third element, every other category is a 2-tuple. We preserve that exactly and
    fill ``expected_card=None`` when absent. Never imports ``eval50`` (which would
    ``from scope_bot import ScopeBot`` and load the model)."""
    assigns = _module_assignments(path)
    if "SCEN" not in assigns:
        raise ValueError(f"source_fingerprint: {path} defines no top-level SCEN")
    try:
        rows = ast.literal_eval(assigns["SCEN"])
    except (ValueError, SyntaxError) as exc:
        raise ValueError(
            f"source_fingerprint: SCEN in {path} is not a literal ({exc})"
        ) from exc
    if not isinstance(rows, (list, tuple)):
        raise ValueError(
            f"source_fingerprint: SCEN in {path} is {type(rows).__name__}, not a list"
        )
    cases: list[EvalCase] = []
    for index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            raise ValueError(
                f"source_fingerprint: SCEN[{index}] is malformed: {row!r}"
            )
        category, query = row[0], row[1]
        expected = row[2] if len(row) > 2 else None
        if category not in _EVAL_CATEGORIES:
            raise ValueError(
                f"source_fingerprint: SCEN[{index}] unknown category {category!r}"
            )
        if not isinstance(query, str):
            raise ValueError(
                f"source_fingerprint: SCEN[{index}] query is not a string"
            )
        if expected is not None and not isinstance(expected, str):
            raise ValueError(
                f"source_fingerprint: SCEN[{index}] expected_card is not a string/None"
            )
        cases.append(
            EvalCase(category=category, query=query, expected_card=expected)
        )
    return tuple(cases)


def eval_suite_sha256(cases: Sequence[EvalCase]) -> str:
    """Structure-only digest of the eval suite, matching the reconciled council value
    03a6f1db...937461. Each case serializes to its original ``SCEN`` tuple shape:
    ``[category, query]`` plus ``[expected_card]`` only when a card is present, so a
    reordering or a changed query changes the digest but formatting never does."""
    payload: list[list[str]] = []
    for case in cases:
        row = [case.category, case.query]
        if case.expected_card is not None:
            row.append(case.expected_card)
        payload.append(row)
    return sha256_bytes(canonical_json(payload))
