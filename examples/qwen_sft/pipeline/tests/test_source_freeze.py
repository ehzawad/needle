"""Source-freeze tests — the supply-chain integrity gate.

Real-world analog: in-toto / Sigstore source attestation verification and lockfile
integrity checks in CI. These assert that the bytes the pipeline treats as frozen
(``scope_bot.py``, ``scope_policy.py``, ``eval50.py``) still hash to exactly the
digests declared in ``config.ci.json``, and that the eval suite parsed straight out
of ``eval50.py`` is the exact 50-case, 20/15/8/5/2 suite the release floor assumes.

All extraction is by AST — nothing here imports ``scope_bot`` or ``eval50`` (which
would pull in torch and load the model).
"""
from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path

# Make the example root importable so ``pipeline.*`` resolves regardless of how the
# test runner was invoked (parents: [0]=tests, [1]=pipeline, [2]=example root).
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.source_fingerprint import (  # noqa: E402
    eval_suite_sha256,
    load_eval50_cases,
    read_scope_constants,
    sha256_file,
)

_PKG = Path(__file__).resolve().parents[1]

# The reconciled council value for the structure-only eval-suite digest. This is NOT
# the whole-file hash of eval50.py (that is config.frozen.eval_suite_sha256); it is
# the canonical digest of the parsed 50 cases used by the candidate manifest.
_CANONICAL_SUITE_SHA256 = (
    "03a6f1db1b594d7b6fd2b12d23bf08d03ebce0e46e0d13abffa659bdab937461"
)
_COMBINED_PROMPT_SHA256 = (
    "04290adb67c40ce1b390d464ea2ead523d2342a4f12fa761a1b9b3928d5ad928"
)
_EXPECTED_CATEGORY_COUNTS = {
    "in_scope": 20,
    "hard_ood": 15,
    "far_ood": 8,
    "ambiguous": 5,
    "adversarial": 2,
}


class SourceFreezeTest(unittest.TestCase):
    def _config(self) -> dict:
        return json.loads((_PKG / "config.ci.json").read_text(encoding="utf-8"))

    def test_frozen_source_hashes_match_config(self) -> None:
        frozen = self._config()["frozen"]
        self.assertEqual(
            sha256_file(_ROOT / "scope_bot.py"), frozen["scope_bot_sha256"]
        )
        self.assertEqual(
            sha256_file(_ROOT / "scope_policy.py"), frozen["scope_policy_sha256"]
        )
        # The config freezes eval_suite_sha256 as the whole-file digest of eval50.py.
        self.assertEqual(
            sha256_file(_ROOT / "eval50.py"), frozen["eval_suite_sha256"]
        )

    def test_load_eval50_returns_fifty_cases(self) -> None:
        cases = load_eval50_cases(_ROOT / "eval50.py")
        self.assertEqual(len(cases), 50)

    def test_eval50_category_counts(self) -> None:
        cases = load_eval50_cases(_ROOT / "eval50.py")
        counts = Counter(case.category for case in cases)
        self.assertEqual(dict(counts), _EXPECTED_CATEGORY_COUNTS)

    def test_eval50_expected_card_presence(self) -> None:
        # Only in-scope rows carry an expected card; every other category has none.
        for case in load_eval50_cases(_ROOT / "eval50.py"):
            if case.category == "in_scope":
                self.assertIsNotNone(case.expected_card, msg=case.query)
            else:
                self.assertIsNone(case.expected_card, msg=case.query)

    def test_canonical_eval_suite_sha256(self) -> None:
        cases = load_eval50_cases(_ROOT / "eval50.py")
        self.assertEqual(eval_suite_sha256(cases), _CANONICAL_SUITE_SHA256)

    def test_both_configs_agree_on_frozen_hashes(self) -> None:
        # A re-freeze (e.g. editing scope_policy.py) must update ci AND demo in lockstep.
        ci = json.loads((_PKG / "config.ci.json").read_text(encoding="utf-8"))["frozen"]
        demo = json.loads((_PKG / "config.demo.json").read_text(encoding="utf-8"))["frozen"]
        self.assertEqual(ci, demo, msg="ci and demo must freeze identical sources")
        for key, rel in [
            ("scope_bot_sha256", "scope_bot.py"),
            ("scope_policy_sha256", "scope_policy.py"),
            ("eval_suite_sha256", "eval50.py"),
        ]:
            self.assertEqual(sha256_file(_ROOT / rel), demo[key], msg=key)

    def test_scope_constants_extracted_by_ast(self) -> None:
        frozen = self._config()["frozen"]
        fp = read_scope_constants(_ROOT / "scope_bot.py")
        self.assertEqual(fp.model_id, frozen["model_id"])
        self.assertEqual(fp.revision, frozen["model_revision"])
        self.assertTrue(fp.judge_sys.startswith("You are a strict SCOPE GATE"))
        self.assertTrue(fp.gen_sys.startswith("You are the Northwind Bank"))
        self.assertEqual(fp.combined_prompt_sha256, _COMBINED_PROMPT_SHA256)


if __name__ == "__main__":
    unittest.main()
