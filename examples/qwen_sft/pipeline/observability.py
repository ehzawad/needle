"""Telemetry, exposition metrics, static dashboard, and conservative drift alerts.

Real-world analog: Prometheus + Grafana, reduced to files. Structured events append to
a JSONL stream (``emit_event`` — the append-only event log a scraper/Loki would ingest);
counters and SLIs are rendered in the Prometheus text-exposition format
(``write_prometheus_text`` — what a node's textfile collector reads); a self-contained
static HTML page stands in for a Grafana dashboard (``render_dashboard``); and drift is
an ALERT only, never a promotion signal (``detect_drift``).

Deliberately humble on statistics: this demo never claims a difference is 'significant',
never promotes on drift, and refuses to alert without enough samples. Safety feedback
(handled by the release controller) may trip the circuit; ordinary drift may not.
"""
from __future__ import annotations

import html
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "emit_event",
    "write_prometheus_text",
    "detect_drift",
    "load_drift_alerts",
    "render_dashboard",
]

# The stable event schema every emitted record conforms to. Missing keys default here so
# downstream readers can rely on the shape regardless of the caller.
_STABLE_FIELDS: tuple[str, ...] = (
    "schema_version",
    "event_id",
    "ts",
    "run_id",
    "stage",
    "event",
    "status",
    "artifact_id",
    "backend",
    "cache_key",
    "duration_ms",
    "device_uuid",
    "counts",
    "error_code",
)

_VALID_DISPOSITIONS = frozenset({"ANSWER", "CLARIFY", "ABSTAIN"})


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_event_id() -> str:
    import hashlib

    seed = os.urandom(16) + str(datetime.now(timezone.utc).timestamp()).encode("utf-8")
    return "sha256:" + hashlib.sha256(seed).hexdigest()


def _event_defaults() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event_id": None,
        "ts": None,
        "run_id": None,
        "stage": None,
        "event": None,
        "status": None,
        "artifact_id": None,
        "backend": None,
        "cache_key": None,
        "duration_ms": None,
        "device_uuid": None,
        "counts": {},
        "error_code": None,
    }


def emit_event(path: Path, event: Mapping[str, Any]) -> None:
    """Append one normalized event as a JSON line to the append-only stream at ``path``.

    The record always carries every stable field (defaulted when the caller omits it),
    plus any extra keys the caller supplied. ``event_id`` / ``ts`` are generated when
    absent so each line is self-identifying and time-stamped.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = _event_defaults()
    for key, value in dict(event).items():
        record[key] = value
    if not record.get("event_id"):
        record["event_id"] = _new_event_id()
    if not record.get("ts"):
        record["ts"] = _utcnow()
    if record.get("counts") is None:
        record["counts"] = {}
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _fmt_metric_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


def write_prometheus_text(path: Path, metrics: Mapping[str, int | float]) -> None:
    """Render ``metrics`` in the Prometheus text-exposition format, written atomically.

    Keys may include label syntax (``name{label="v"}``); the base name (before ``{``)
    gets a single ``# HELP`` / ``# TYPE`` header, with ``_total`` metrics typed as
    counters and everything else as gauges. Sorted output keeps the file byte-stable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    seen_base: set[str] = set()
    for name in sorted(metrics):
        base = name.split("{", 1)[0]
        if base not in seen_base:
            mtype = "counter" if base.endswith("_total") else "gauge"
            lines.append(f"# HELP {base} scope pipeline metric {base}")
            lines.append(f"# TYPE {base} {mtype}")
            seen_base.add(base)
        lines.append(f"{name} {_fmt_metric_value(metrics[name])}")
    text = ("\n".join(lines) + "\n") if lines else ""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _proportions(
    records: Sequence[Mapping[str, Any]], key: str, valid: frozenset[str] | None
) -> tuple[dict[str, float], int]:
    """Return ``(value -> proportion, total)`` for a field across records.

    Records whose field is missing/None (or outside ``valid`` when given) are ignored so
    the denominator counts only classifiable turns.
    """
    counts: dict[str, int] = {}
    total = 0
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        if valid is not None and value not in valid:
            continue
        counts[value] = counts.get(value, 0) + 1
        total += 1
    if total == 0:
        return {}, 0
    return {k: v / total for k, v in counts.items()}, total


def detect_drift(
    baseline: Sequence[Mapping[str, Any]],
    recent: Sequence[Mapping[str, Any]],
    *,
    min_samples: int,
    max_rate_delta: float,
) -> tuple[Mapping[str, Any], ...]:
    """Return conservative, non-significant drift alerts (possibly empty).

    Refuses to say anything without at least ``min_samples`` classifiable turns in BOTH
    the baseline and the recent window. Compares ANSWER/CLARIFY/ABSTAIN proportions and
    per-card proportions to the release baseline and alerts only when an absolute rate
    delta exceeds ``max_rate_delta``. These are informational: they never gate promotion.
    """
    alerts: list[Mapping[str, Any]] = []

    base_disp, base_total = _proportions(baseline, "disposition", _VALID_DISPOSITIONS)
    recent_disp, recent_total = _proportions(recent, "disposition", _VALID_DISPOSITIONS)
    if base_total < min_samples or recent_total < min_samples:
        return ()  # not enough evidence to responsibly say anything.

    def _emit(dimension: str, key: str, b_rate: float, r_rate: float, b_n: int, r_n: int) -> None:
        delta = abs(r_rate - b_rate)
        if delta > max_rate_delta:
            alerts.append(
                {
                    "kind": "drift",
                    "dimension": dimension,
                    "key": key,
                    "baseline_rate": round(b_rate, 4),
                    "recent_rate": round(r_rate, 4),
                    "delta": round(delta, 4),
                    "threshold": max_rate_delta,
                    "samples_baseline": b_n,
                    "samples_recent": r_n,
                    "note": (
                        "informational drift alert; NOT a statistical-significance "
                        "claim; does not gate promotion"
                    ),
                }
            )

    for disposition in sorted(set(base_disp) | set(recent_disp)):
        _emit(
            "disposition",
            disposition,
            base_disp.get(disposition, 0.0),
            recent_disp.get(disposition, 0.0),
            base_total,
            recent_total,
        )

    base_card, base_card_total = _proportions(baseline, "card_id", None)
    recent_card, recent_card_total = _proportions(recent, "card_id", None)
    if base_card_total >= min_samples and recent_card_total >= min_samples:
        for card_id in sorted(set(base_card) | set(recent_card)):
            _emit(
                "card",
                card_id,
                base_card.get(card_id, 0.0),
                recent_card.get(card_id, 0.0),
                base_card_total,
                recent_card_total,
            )

    return tuple(alerts)


def load_drift_alerts(
    log_path: Path | str | None,
    *,
    min_samples: int,
    max_rate_delta: float,
) -> list[Mapping[str, Any]]:
    """Load conservative drift alerts from the configured live feedback log.

    The SINGLE drift loader shared by the CLI ``dashboard`` command and the DAG's
    observability writer, so both render identical alerts from the same CONFIGURED path
    (never ``feedback_log.read_sessions``'s CWD-relative default). Reads the served turns,
    splits them older-half vs newer-half by timestamp, and delegates to
    :func:`detect_drift`. Returns ``[]`` when no path is configured or there are fewer
    than ``min_samples`` classifiable turns per window — the honest behavior at demo
    volume. Drift is informational only; it never gates promotion.
    """
    if not log_path:
        return []
    from feedback_log import read_sessions

    sessions = read_sessions(Path(log_path))
    turns = [t for session in sessions.values() for t in session if t.get("disposition")]
    turns.sort(key=lambda t: t.get("ts", ""))
    mid = len(turns) // 2
    return list(
        detect_drift(
            turns[:mid],
            turns[mid:],
            min_samples=min_samples,
            max_rate_delta=max_rate_delta,
        )
    )


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _evidence_rows(evidence: Sequence[Mapping[str, Any]]) -> str:
    if not evidence:
        return '<tr><td colspan="4" class="muted">no evidence recorded</td></tr>'
    rows: list[str] = []
    for item in evidence:
        kind = _esc(item.get("kind"))
        passed = item.get("passed")
        badge = "pass" if passed else ("fail" if passed is False else "na")
        label = {"pass": "PASS", "fail": "FAIL", "na": "—"}[badge]
        metrics = item.get("metrics") or {}
        summary = ", ".join(f"{_esc(k)}={_esc(v)}" for k, v in dict(metrics).items())
        rows.append(
            f'<tr><td>{kind}</td>'
            f'<td><span class="badge {badge}">{label}</span></td>'
            f'<td>{_esc(item.get("artifact_id"))}</td>'
            f'<td class="metrics">{summary}</td></tr>'
        )
    return "\n".join(rows)


def _alert_items(alerts: Sequence[Mapping[str, Any]]) -> str:
    if not alerts:
        return '<li class="muted">no active alerts</li>'
    items: list[str] = []
    for alert in alerts:
        dim = _esc(alert.get("dimension") or alert.get("kind"))
        key = _esc(alert.get("key"))
        delta = _esc(alert.get("delta"))
        threshold = _esc(alert.get("threshold"))
        items.append(
            f"<li><strong>{dim}:{key}</strong> "
            f"delta {delta} &gt; {threshold} "
            f'<span class="muted">(informational — does not gate promotion)</span></li>'
        )
    return "\n".join(items)


def render_dashboard(
    output: Path,
    *,
    release: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    alerts: Sequence[Mapping[str, Any]],
) -> None:
    """Write a self-contained static HTML dashboard (no external assets) to ``output``.

    Renders the current release pointer, the promotion evidence (offline/shadow/canary),
    and any conservative drift alerts. All dynamic values are HTML-escaped.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    generated = _utcnow()
    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scope pipeline dashboard</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         margin: 0; padding: 2rem; line-height: 1.5;
         background: Canvas; color: CanvasText; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 0.25rem; }}
  .muted {{ opacity: 0.65; }}
  .grid {{ display: grid; gap: 1.5rem; max-width: 60rem; }}
  .card {{ border: 1px solid color-mix(in srgb, CanvasText 18%, transparent);
          border-radius: 12px; padding: 1.25rem; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ text-align: left; padding: 0.4rem 0.6rem;
           border-bottom: 1px solid color-mix(in srgb, CanvasText 12%, transparent);
           vertical-align: top; }}
  .metrics {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
             font-size: 0.85rem; word-break: break-word; }}
  .badge {{ font-size: 0.75rem; font-weight: 700; padding: 0.1rem 0.5rem;
           border-radius: 999px; }}
  .badge.pass {{ background: #1a7f37; color: #fff; }}
  .badge.fail {{ background: #b32424; color: #fff; }}
  .badge.na {{ background: color-mix(in srgb, CanvasText 20%, transparent); }}
  dl {{ display: grid; grid-template-columns: max-content 1fr; gap: 0.25rem 1rem;
       margin: 0; }}
  dt {{ font-weight: 600; }}
  dd {{ margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
       font-size: 0.85rem; word-break: break-word; }}
  ul {{ margin: 0; padding-left: 1.2rem; }}
</style>
</head>
<body>
<div class="grid">
  <header>
    <h1>Scope pipeline dashboard</h1>
    <p class="muted">generated {_esc(generated)} — static snapshot, no live scraping</p>
  </header>

  <section class="card">
    <h2>Current release</h2>
    <dl>
      <dt>channel</dt><dd>{_esc(release.get("channel", "CURRENT"))}</dd>
      <dt>artifact_id</dt><dd>{_esc(release.get("artifact_id"))}</dd>
      <dt>previous</dt><dd>{_esc(release.get("previous_artifact_id"))}</dd>
      <dt>backend</dt><dd>{_esc(release.get("backend"))}</dd>
      <dt>actor</dt><dd>{_esc(release.get("actor"))}</dd>
      <dt>updated_at</dt><dd>{_esc(release.get("updated_at"))}</dd>
    </dl>
  </section>

  <section class="card">
    <h2>Promotion evidence</h2>
    <table>
      <thead><tr><th>stage</th><th>result</th><th>artifact</th><th>metrics</th></tr></thead>
      <tbody>
{_evidence_rows(evidence)}
      </tbody>
    </table>
  </section>

  <section class="card">
    <h2>Alerts</h2>
    <ul>
{_alert_items(alerts)}
    </ul>
  </section>
</div>
</body>
</html>
"""
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(doc, encoding="utf-8")
    os.replace(tmp, output)
