"""The mission-critical three-tier selective router — the culmination.

Ties together everything measured across the councils:
  E5 embedding -> intent probe + fused OOD head -> AUTO / CLARIFY / ESCALATE.

Decision per query (never a silent rejection):
  ESCALATE : fused OOD probability high  -> likely doppelganger/OOD -> human/fallback
  CLARIFY  : in-scope, but sibling margin low (or a required slot missing)
             -> ask "did you mean A or B?" using the candidate intents' CANONICAL questions
  AUTO     : in-scope AND clear winner AND hard-gates pass -> return the fixed answer

Two thresholds are calibrated on validation:
  tau_ood    : escalate boundary (set from a target OOD-rejection / cost)
  tau_margin : auto-vs-clarify boundary, set so the AUTO region hits a target accuracy

In a Cactus tool-calling runtime this class is the ORCHESTRATOR: it emits either the
answer tool call, an `ask_clarification` tool call, or an escalate signal. Cactus/Needle
is NOT on the routing path for this fixed-answer task.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression

OOS = 150


def load(data, split):
    return [json.loads(l) for l in (Path(data) / f"records_{split}.jsonl").read_text().splitlines()]


class SelectivePolicy:
    def __init__(self, data="data", cache_model="intfloat/multilingual-e5-large-instruct",
                 target_ood_recall=0.90, auto_acc_target=0.99):
        self.data = Path(data)
        self.cache_model = cache_model
        self._embedder = None
        cat = json.loads((self.data / "catalog.json").read_text())
        self.intent_ids = [c["id"] for c in cat]
        self.answer_map = json.loads((self.data / "answer_map.json").read_text())
        idx = {n: i for i, n in enumerate(self.intent_ids)}
        tr, va = load(data, "train"), load(data, "validation")
        Xtr = self.embed_split("train", tr)
        Xva = self.embed_split("val", va)
        lab = lambda R: np.array([OOS if r["is_oos"] else idx[r["intent_id"]] for r in R])
        ytr, yva = lab(tr), lab(va)
        in_tr = ytr < OOS
        self.Xid, self.yid = Xtr[in_tr], ytr[in_tr]

        # intent probe + centroids + canonical (medoid) questions
        self.probe = LogisticRegression(C=100, max_iter=4000).fit(self.Xid, self.yid)
        self.classes = self.probe.classes_
        tr_txt = [r["query"] for r in tr]
        cents, medoid_txt = [], []
        for c in range(OOS):
            rows = np.where(ytr == c)[0]
            V = Xtr[rows]
            cen = V.mean(0); cen /= np.linalg.norm(cen) + 1e-12
            cents.append(cen)
            medoid_txt.append(tr_txt[rows[int(np.argmax(V @ cen))]])
        self.C = np.stack(cents).astype(np.float32)
        self.canonical = medoid_txt

        # Mahalanobis whitening (tied + background) for MD / relative-MD
        mus = np.stack([self.Xid[self.yid == c].mean(0) for c in range(OOS)]).astype(np.float32)
        self.mus = mus
        self.M = np.linalg.cholesky(LedoitWolf().fit(self.Xid - mus[self.yid]).precision_).astype(np.float32)
        self.musw = mus @ self.M
        self.mu0 = self.Xid.mean(0).astype(np.float32)
        self.M0 = np.linalg.cholesky(LedoitWolf().fit(self.Xid - self.mu0).precision_).astype(np.float32)
        self.mu0w = self.mu0 @ self.M0
        self.Tid = self.Xid  # kNN reference

        # fused OOD head fit on validation features
        Fva = self._features(Xva)
        self.ood = LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced").fit(
            Fva, (yva == OOS).astype(int))

        # calibrate tau_ood at target OOD recall on val OOD
        pva = self.ood.predict_proba(Fva)[:, 1]
        self.tau_ood = float(np.quantile(pva[yva == OOS], 1 - target_ood_recall))

        # calibrate tau_margin: among val in-scope that pass the OOD gate, pick the
        # smallest margin whose AUTO region reaches auto_acc_target accuracy
        in_va = yva < OOS
        pred_va = self.classes[self.probe.predict_proba(Xva).argmax(1)]
        margin_va = self._margin(Xva)
        keep = in_va & (pva < self.tau_ood)
        m, correct = margin_va[keep], (pred_va[keep] == yva[keep])
        self.tau_margin = self._pick_margin(m, correct, auto_acc_target)
        self.target_ood_recall = target_ood_recall
        self.auto_acc_target = auto_acc_target

    def embed_split(self, split, records):
        """Load cached query embeddings for a split, embedding on-demand if absent.

        Self-contained: the pipeline needs only convert_clinc.py (for data/) and an
        E5 model — no separate index/embedding-build step. The cache is regenerable
        and git-ignored.
        """
        cdir = Path("cache") / self.cache_model.replace("/", "_")
        p = cdir / f"{split}.npz"
        if p.exists():
            return np.load(p)["X"].astype(np.float32)
        if self._embedder is None:
            from embedder import Embedder
            self._embedder = Embedder(self.cache_model)
        X = self._embedder.encode_queries([r["query"] for r in records]).astype(np.float32)
        cdir.mkdir(parents=True, exist_ok=True)
        np.savez(p, X=X)
        return X

    # ---- feature pipeline (higher fused prob = more OOD) ----
    def _probe_stats(self, X):
        P = self.probe.predict_proba(X)
        s = np.sort(P, 1)
        pred = self.classes[P.argmax(1)]
        return P.max(1), -(P * np.log(P + 1e-12)).sum(1), s[:, -1] - s[:, -2], pred

    def _margin(self, X):
        P = self.probe.predict_proba(X); s = np.sort(P, 1)
        return s[:, -1] - s[:, -2]

    def _md_rmd(self, X):
        Xw = X @ self.M
        d2 = ((Xw[:, None, :] - self.musw[None, :, :]) ** 2).sum(-1)
        md = d2.min(1)
        bg = (((X @ self.M0) - self.mu0w) ** 2).sum(1)
        return md, md - bg

    def _knn(self, X, k=5):
        out = []
        for i in range(0, len(X), 512):
            sims = X[i:i+512] @ self.Tid.T
            out.append(1.0 - np.sort(sims, 1)[:, -k:].mean(1))
        return np.concatenate(out)

    def _features(self, X):
        p1, ent, mgn, _ = self._probe_stats(X)
        S = X @ self.C.T
        s = np.sort(S, 1)
        md, rmd = self._md_rmd(X)
        return np.column_stack([p1, ent, mgn, s[:, -1], s[:, -1] - s[:, -2], self._knn(X), md, rmd])

    @staticmethod
    def _pick_margin(margin, correct, target):
        order = np.argsort(-margin)  # high margin first
        m_sorted, c_sorted = margin[order], correct[order]
        cum_acc = np.cumsum(c_sorted) / (np.arange(len(c_sorted)) + 1)
        ok = np.where(cum_acc >= target)[0]
        if len(ok) == 0:
            return float(margin.max())  # target unattainable -> AUTO only the single best
        last = ok[-1]
        return float(m_sorted[last])

    def _clarify_menu(self, order_i):
        cands = [self.intent_ids[int(c)] for c in order_i[:3]]
        return [{"intent": c, "question": self.canonical[self.intent_ids.index(c)]} for c in cands]

    # ---- the decision ----
    def decide_batch(self, X, queries, missing_slot=None, scope_gate=None):
        """scope_gate(query, top_intent) -> 'ESCALATE' | 'CLARIFY' | None.

        This is the registry COLLISION-RULE hook (council-4). An exact doppelgänger
        (identical text to an in-scope query) has HIGH confidence and will otherwise
        reach AUTO, so a known-collision override must run BEFORE the confidence
        logic. Populate it from your intent registry (e.g. card-PIN vs SIM-PIN via a
        required scope-defining slot). Defaults to no-op.
        """
        pood = self.ood.predict_proba(self._features(X))[:, 1]
        P = self.probe.predict_proba(X)
        order = np.argsort(-P, 1)
        pred = self.classes[order[:, 0]]
        margin = P[np.arange(len(P)), order[:, 0]] - P[np.arange(len(P)), order[:, 1]]
        out = []
        for i in range(len(X)):
            slot_missing = bool(missing_slot[i]) if missing_slot is not None else False
            top_intent = self.intent_ids[int(order[i, 0])]
            override = scope_gate(queries[i], top_intent) if scope_gate else None
            if override == "ESCALATE":
                out.append({"disposition": "ESCALATE", "reason": "known out-of-scope collision (registry rule)",
                            "p_ood": float(pood[i]), "answer": None})
            elif override == "CLARIFY":
                out.append({"disposition": "CLARIFY", "reason": "scope-defining slot unresolved (registry rule)",
                            "p_ood": float(pood[i]), "margin": float(margin[i]),
                            "candidates": self._clarify_menu(order[i]), "answer": None})
            elif pood[i] >= self.tau_ood:
                out.append({"disposition": "ESCALATE", "reason": "low in-scope support (likely OOD)",
                            "p_ood": float(pood[i]), "answer": None})
            elif margin[i] < self.tau_margin or slot_missing:
                cands = [self.intent_ids[int(c)] for c in order[i, :3]]
                menu = [{"intent": c, "question": self.canonical[self.intent_ids.index(c)]} for c in cands]
                out.append({"disposition": "CLARIFY",
                            "reason": "missing required detail" if slot_missing else "sibling intents too close",
                            "p_ood": float(pood[i]), "margin": float(margin[i]),
                            "candidates": menu, "answer": None})
            else:
                intent = self.intent_ids[int(order[i, 0])]
                out.append({"disposition": "AUTO", "intent": intent,
                            "answer": self.answer_map.get(intent),
                            "p_ood": float(pood[i]), "margin": float(margin[i])})
        return out

    @staticmethod
    def to_tool_call(d):
        """Render a disposition as the tool call an orchestrator would emit to Cactus."""
        if d["disposition"] == "AUTO":
            return {"name": "answer", "arguments": {"intent": d["intent"]}}
        if d["disposition"] == "CLARIFY":
            return {"name": "ask_clarification",
                    "arguments": {"options": [c["question"] for c in d["candidates"]], "allow_none": True}}
        return {"name": "escalate_to_human", "arguments": {"reason": d["reason"]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="intfloat/multilingual-e5-large-instruct")
    ap.add_argument("--target-ood-recall", type=float, default=0.90)
    ap.add_argument("--auto-acc-target", type=float, default=0.99)
    args = ap.parse_args()

    pol = SelectivePolicy(args.data, args.model, args.target_ood_recall, args.auto_acc_target)
    print(f"calibrated: tau_ood={pol.tau_ood:.4f} (target {args.target_ood_recall:.0%} OOD-reject), "
          f"tau_margin={pol.tau_margin:.4f} (AUTO acc target {args.auto_acc_target:.0%})")

    # evaluate on sealed test
    te = load(args.data, "test")
    Xte = pol.embed_split("test", te)
    idx = {n: i for i, n in enumerate(pol.intent_ids)}
    yte = np.array([OOS if r["is_oos"] else idx[r["intent_id"]] for r in te])
    disp = pol.decide_batch(Xte, [r["query"] for r in te])

    is_mask = yte < OOS
    tiers = {"AUTO": 0, "CLARIFY": 0, "ESCALATE": 0}
    auto_correct = auto_total = auto_oos = 0
    for i, d in enumerate(disp):
        tiers[d["disposition"]] += 1
        if d["disposition"] == "AUTO":
            if is_mask[i]:
                auto_total += 1
                auto_correct += (d["intent"] == pol.intent_ids[int(yte[i])])
            else:
                auto_oos += 1
    n = len(disp)
    print(f"\nsealed test ({int(is_mask.sum())} in-scope + {int((~is_mask).sum())} OOD):")
    for t in ("AUTO", "CLARIFY", "ESCALATE"):
        print(f"  {t:9} {tiers[t]:5}  ({tiers[t]/n:5.1%} of traffic)")
    print(f"\n  AUTO in-scope accuracy : {auto_correct}/{auto_total} = {auto_correct/max(auto_total,1):.4f}")
    print(f"  AUTO OOD leak (bad!)   : {auto_oos}  ({auto_oos/max(tiers['AUTO'],1):.4%} of AUTO)")
    # how many legitimate queries are recovered vs escalated
    is_auto = sum(1 for i, d in enumerate(disp) if is_mask[i] and d["disposition"] == "AUTO")
    is_clar = sum(1 for i, d in enumerate(disp) if is_mask[i] and d["disposition"] == "CLARIFY")
    is_esc = sum(1 for i, d in enumerate(disp) if is_mask[i] and d["disposition"] == "ESCALATE")
    print(f"\n  legitimate in-scope handling: AUTO {is_auto/int(is_mask.sum()):.1%} | "
          f"CLARIFY {is_clar/int(is_mask.sum()):.1%} | ESCALATE {is_esc/int(is_mask.sum()):.1%}")

    print("\n  --- sample dispositions + the tool call the orchestrator emits ---")
    import numpy as _np
    rng = _np.random.RandomState(3)
    for t in ("AUTO", "CLARIFY", "ESCALATE"):
        cand = [i for i, d in enumerate(disp) if d["disposition"] == t]
        for i in rng.choice(cand, size=min(2, len(cand)), replace=False):
            d = disp[i]
            print(f"\n  [{t}] Q: {te[i]['query'][:80]}  (gold={'OOD' if yte[i]==OOS else pol.intent_ids[int(yte[i])]})")
            if t == "CLARIFY":
                print("        menu: " + " | ".join(c["question"][:45] for c in d["candidates"][:2]))
            print("        tool_call: " + json.dumps(pol.to_tool_call(d)))


if __name__ == "__main__":
    main()
