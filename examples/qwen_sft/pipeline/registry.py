"""Immutable model registry — content-addressed blobs, candidates, and evidence.

Real-world analog: an MLflow / Weights & Biases Model Registry layered over an
OCI-style content-addressed object store (like a container registry's ``blobs/sha256``
tree or DVC/lakeFS). It teaches the production ideas that matter without any server,
database, or network:

  * every input byte-string is stored ONCE, keyed by its own SHA-256 (a blob);
  * a candidate is an IMMUTABLE artifact directory: a signed-by-hash manifest plus a
    ``files/`` copy of its payload, published atomically via ``os.replace`` so a reader
    never observes a half-written artifact;
  * the ``artifact_id`` IS the SHA-256 of the candidate's identity (payload + runtime +
    evaluation contract + lineage), so identity is reproducible and tamper-evident;
  * evaluation ``evidence`` is likewise content-addressed and carries its ``backend``
    (``mock``/``real``) permanently, which the release controller uses to keep CI mock
    evidence out of a demo promotion;
  * ``verify_candidate`` re-hashes every referenced blob and the manifest itself and
    fails closed on any corruption — the integrity check serving relies on before load.

Pure standard library. Never imports torch/transformers/scope_bot; it only moves and
hashes bytes. The canonical-JSON and SHA-256 helpers are shared with the rest of the
pipeline (via ``source_fingerprint``) so hashes computed here match everywhere.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pipeline.contracts import BlobRef
from pipeline.source_fingerprint import canonical_json, sha256_bytes

SCHEMA_VERSION = 1
_CANDIDATE_KIND = "scope_bot_candidate"
_IDENTITY_KEYS = ("payload", "runtime", "evaluation_contract", "lineage")
_EVIDENCE_KINDS = ("offline_eval", "shadow", "canary")
_ENVIRONMENTS = ("ci", "demo")


class RegistryError(RuntimeError):
    """Raised on a registry contract violation or detected corruption (fail-closed)."""


# ---------------------------------------------------------------------------
# Durable, atomic filesystem primitives (shared with release.py, same lane).
# ---------------------------------------------------------------------------


def fsync_dir(path: Path) -> None:
    """fsync a directory so a rename/replace inside it is durable across a crash.

    Some filesystems/platforms reject directory fsync; that is non-fatal here.
    """
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_all(fd: int, data: bytes) -> None:
    """Write EVERY byte of ``data`` to ``fd``, looping over short ``os.write`` returns.

    ``os.write`` is permitted to write fewer bytes than requested (notably on pipes, and
    in principle on any fd); a single call is not guaranteed to persist the whole buffer.
    Callers that append audit records or publish durable files must not silently drop the
    tail, so this loops until the buffer is fully written and fails closed if a write
    makes no progress.
    """
    view = memoryview(data)
    total = 0
    length = len(view)
    while total < length:
        written = os.write(fd, view[total:])
        if written <= 0:
            raise RegistryError("write_all: os.write made no progress")
        total += written


def atomic_write_bytes(path: Path, data: bytes, *, read_only: bool = False) -> None:
    """Atomically publish ``data`` at ``path``: write a same-directory temp file,
    ``fsync`` it, ``os.replace`` it into place, then ``fsync`` the directory.

    A reader therefore sees either the old bytes or the complete new bytes, never a
    torn write. When ``read_only`` is set the published file is left mode ``0o444`` so
    immutable content (blobs, manifests, candidate files) resists in-place rewrites.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}"
    try:
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            write_all(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        if read_only:
            os.chmod(str(tmp), 0o444)
        os.replace(str(tmp), str(path))
    except BaseException:
        try:
            os.unlink(str(tmp))
        except OSError:
            pass
        raise
    fsync_dir(path.parent)


def dumps_bytes(obj: object) -> bytes:
    """Human-readable, deterministic JSON bytes (sorted keys) with a trailing newline.

    On-disk formatting is independent of the content-address: every hash the registry
    checks is recomputed with :func:`canonical_json` after parsing, so pretty-printing
    here never changes an ``artifact_id``, ``evidence_id``, or ``manifest_sha256``.
    """
    return (
        json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string ending in ``Z``."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def with_sha256_prefix(value: object | None) -> str | None:
    """Normalize an id to the canonical ``sha256:<hex>`` form (``None`` passes through)."""
    if value is None:
        return None
    text = str(value)
    return text if text.startswith("sha256:") else f"sha256:{text}"


# ---------------------------------------------------------------------------
# JSON-normalization and blob discovery helpers.
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """Convert an identity/evidence value into plain JSON types deterministically.

    A :class:`BlobRef` becomes ``{"sha256", "bytes", "media_type"}`` and tuples become
    lists, so the structure equals exactly what a JSON round-trip yields on read — which
    is what makes the recomputed ``artifact_id``/``evidence_id`` stable.
    """
    if isinstance(value, BlobRef):
        return {"sha256": value.sha256, "bytes": value.bytes, "media_type": value.media_type}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int, float)):
        return value
    raise RegistryError(
        f"registry: value of type {type(value).__name__} is not JSON-serializable"
    )


def _iter_blobrefs(node: Any) -> list[dict[str, Any]]:
    """Collect every BlobRef-shaped leaf (``sha256`` hex, ``bytes`` int, ``media_type``)
    reachable in a payload tree, without recursing into a matched leaf."""
    found: list[dict[str, Any]] = []
    if isinstance(node, Mapping):
        if (
            isinstance(node.get("sha256"), str)
            and isinstance(node.get("bytes"), int)
            and not isinstance(node.get("bytes"), bool)
            and "media_type" in node
        ):
            found.append(dict(node))
        else:
            for value in node.values():
                found.extend(_iter_blobrefs(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_iter_blobrefs(value))
    return found


def _guess_media_type(name: str) -> str:
    return "application/json" if name.endswith(".json") else "application/octet-stream"


def _safe_member_name(name: str) -> str:
    if not name or "/" in name or "\\" in name or name in (".", "..") or name.startswith("."):
        raise RegistryError(f"registry: unsafe artifact file name {name!r}")
    return name


class FileRegistry:
    """A file-backed, content-addressed registry rooted at a single state directory.

    The directory layout is the on-disk schema the whole pipeline agrees on::

        <root>/
          blobs/sha256/<digest>
          artifacts/<artifact_id>/manifest.json + files/<name>
          evidence/<evidence_id>.json
          channels/{SHADOW,CANARY,CURRENT}   (written by release.py)
          releases/history.jsonl             (written by release.py)
          circuit.json                       (written by release.py)
          runs/<run_id>/...                  (written by dag.py)
          observability/...                  (written by observability.py)
          .release.lock                      (held by release.py)

    ``environment`` (``ci``/``demo``) is stamped into evidence and is what the release
    controller consults to refuse mock evidence on a demo promotion.
    """

    def __init__(self, root: Path, *, environment: Literal["ci", "demo"]) -> None:
        if environment not in _ENVIRONMENTS:
            raise RegistryError(
                f"registry: environment must be one of {_ENVIRONMENTS}, got {environment!r}"
            )
        self.root = Path(root).resolve()
        self.environment = environment
        self.blobs_dir = self.root / "blobs" / "sha256"
        self.artifacts_dir = self.root / "artifacts"
        self.evidence_dir = self.root / "evidence"
        self.channels_dir = self.root / "channels"
        self.releases_dir = self.root / "releases"
        self.runs_dir = self.root / "runs"
        self.observability_dir = self.root / "observability"
        self.history_path = self.releases_dir / "history.jsonl"
        self.circuit_path = self.root / "circuit.json"
        self.reviews_path = self.root / "reviews.jsonl"
        self.lock_path = self.root / ".release.lock"
        for directory in (
            self.blobs_dir,
            self.artifacts_dir,
            self.evidence_dir,
            self.channels_dir,
            self.releases_dir,
            self.runs_dir,
            self.observability_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    # -- blobs --------------------------------------------------------------

    def put_blob(self, data: bytes, *, media_type: str) -> BlobRef:
        """Store ``data`` under ``blobs/sha256/<digest>`` and return its :class:`BlobRef`.

        Content-addressed and idempotent: re-storing identical bytes returns the same
        ref and never rewrites the file. If a blob with this digest already exists but
        its bytes no longer hash to that digest, the store is corrupt and we fail closed.
        """
        if not isinstance(data, (bytes, bytearray)):
            raise RegistryError("put_blob: data must be bytes")
        payload = bytes(data)
        digest = sha256_bytes(payload)
        path = self.blobs_dir / digest
        if path.exists():
            existing = path.read_bytes()
            if sha256_bytes(existing) != digest:
                raise RegistryError(f"put_blob: existing blob {digest} is corrupted on disk")
        else:
            atomic_write_bytes(path, payload, read_only=True)
        return BlobRef(sha256=digest, bytes=len(payload), media_type=media_type)

    def read_blob(self, ref: BlobRef) -> bytes:
        """Read and integrity-check a blob: its bytes must hash to ``ref.sha256`` and be
        ``ref.bytes`` long, else the read fails closed."""
        path = self.blobs_dir / ref.sha256
        if not path.is_file():
            raise RegistryError(f"read_blob: missing blob {ref.sha256}")
        data = path.read_bytes()
        if len(data) != ref.bytes or sha256_bytes(data) != ref.sha256:
            raise RegistryError(f"read_blob: blob {ref.sha256} failed integrity check")
        return data

    # -- candidates ---------------------------------------------------------

    def artifact_dir(self, artifact_id: str) -> Path:
        """Directory that holds (or would hold) a candidate's manifest and files."""
        return self.artifacts_dir / str(with_sha256_prefix(artifact_id))

    def register_candidate(
        self,
        identity: Mapping[str, Any],
        files: Mapping[str, bytes],
    ) -> Mapping[str, Any]:
        """Publish an immutable candidate and return its manifest mapping.

        ``identity`` must carry ``payload``, ``runtime``, ``evaluation_contract`` and
        ``lineage``; ``artifact_id = "sha256:" + sha256(canonical(those four))`` — it
        excludes timestamps and ids so it is reproducible. ``files`` (e.g.
        ``cards.json``/``exemplars.json``) are stored as blobs AND copied under
        ``files/`` so the artifact directory is self-contained. The whole directory is
        assembled in a private staging path and made visible with a single
        ``os.replace``. Re-registering the same identity is idempotent: the existing
        (verified) manifest is returned and never overwritten.
        """
        core: dict[str, Any] = {}
        for key in _IDENTITY_KEYS:
            if key not in identity:
                raise RegistryError(f"register_candidate: identity missing {key!r}")
            core[key] = _jsonable(identity[key])

        digest = sha256_bytes(canonical_json(core))
        artifact_id = f"sha256:{digest}"
        adir = self.artifact_dir(artifact_id)
        if adir.exists():
            # Immutable: never rewrite. Prove it is intact, then return it unchanged.
            self.verify_candidate(artifact_id)
            return self.load_candidate(artifact_id)

        payload = {str(_safe_member_name(name)): bytes(data) for name, data in files.items()}
        for name, data in payload.items():
            # Every artifact file is also a first-class content-addressed blob.
            self.put_blob(data, media_type=_guess_media_type(name))

        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": _CANDIDATE_KIND,
            "artifact_id": artifact_id,
            "created_at": utc_now_iso(),
            "payload": core["payload"],
            "runtime": core["runtime"],
            "evaluation_contract": core["evaluation_contract"],
            "lineage": core["lineage"],
        }
        manifest["manifest_sha256"] = sha256_bytes(canonical_json(manifest))

        staging = self.artifacts_dir / f".staging-{digest}-{os.getpid()}-{uuid4().hex}"
        try:
            (staging / "files").mkdir(parents=True)
            for name, data in payload.items():
                atomic_write_bytes(staging / "files" / name, data, read_only=True)
            atomic_write_bytes(staging / "manifest.json", dumps_bytes(manifest), read_only=True)
            fsync_dir(staging)
            os.replace(str(staging), str(adir))
        except BaseException:
            _rmtree_quiet(staging)
            raise
        fsync_dir(self.artifacts_dir)
        return manifest

    def load_candidate(self, artifact_id: str) -> Mapping[str, Any]:
        """Return a candidate's manifest mapping (no re-hashing; use verify for that)."""
        manifest_path = self.artifact_dir(artifact_id) / "manifest.json"
        if not manifest_path.is_file():
            raise RegistryError(f"load_candidate: unknown artifact {artifact_id}")
        try:
            manifest = json.loads(manifest_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryError(
                f"load_candidate: manifest for {artifact_id} is unreadable ({exc})"
            ) from exc
        if not isinstance(manifest, dict):
            raise RegistryError(f"load_candidate: manifest for {artifact_id} is not an object")
        return manifest

    def verify_candidate(self, artifact_id: str) -> None:
        """Re-derive and re-hash everything, raising :class:`RegistryError` on any drift.

        Checks, in order: the manifest hash (``manifest_sha256`` over the manifest minus
        itself), the reproduced ``artifact_id`` (over payload+runtime+evaluation_contract
        +lineage), every payload blob (present, right length, right digest), and every
        ``files/`` copy (backed byte-for-byte by an intact blob). This is the gate the
        release controller and serving path run before trusting an artifact.
        """
        expected_id = str(with_sha256_prefix(artifact_id))
        manifest = self.load_candidate(artifact_id)

        stored_manifest_sha = manifest.get("manifest_sha256")
        body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        if sha256_bytes(canonical_json(body)) != stored_manifest_sha:
            raise RegistryError(f"verify_candidate: manifest hash mismatch for {expected_id}")

        missing = [k for k in _IDENTITY_KEYS if k not in manifest]
        if missing:
            raise RegistryError(
                f"verify_candidate: manifest for {expected_id} missing {missing}"
            )
        core = {k: manifest[k] for k in _IDENTITY_KEYS}
        recomputed_id = f"sha256:{sha256_bytes(canonical_json(core))}"
        if recomputed_id != manifest.get("artifact_id") or recomputed_id != expected_id:
            raise RegistryError(
                f"verify_candidate: artifact_id mismatch (stored "
                f"{manifest.get('artifact_id')!r}, recomputed {recomputed_id!r}, "
                f"requested {expected_id!r})"
            )

        for ref in _iter_blobrefs(manifest["payload"]):
            blob_path = self.blobs_dir / ref["sha256"]
            if not blob_path.is_file():
                raise RegistryError(f"verify_candidate: missing blob {ref['sha256']}")
            raw = blob_path.read_bytes()
            if len(raw) != ref["bytes"] or sha256_bytes(raw) != ref["sha256"]:
                raise RegistryError(f"verify_candidate: corrupted blob {ref['sha256']}")

        files_dir = self.artifact_dir(artifact_id) / "files"

        if manifest.get("kind") == _CANDIDATE_KIND:
            # Bind the SERVED files to the artifact's identity: the payload names exactly
            # which blob is the cards KB and which is the exemplar bank, and the served
            # copies MUST be those blobs byte-for-byte. Without this, an identity that
            # hashes cards blob A could ship a files/cards.json holding a different (but
            # individually valid) blob B, and serving would load B while verification
            # trusted A. Any missing/mismatched/unexpected candidate file fails closed.
            payload = manifest.get("payload")
            if not isinstance(payload, Mapping):
                raise RegistryError(
                    f"verify_candidate: manifest for {expected_id} has no payload mapping"
                )
            required_files = {"cards.json": "cards", "exemplars.json": "exemplar_bank"}
            present = (
                {m.name for m in files_dir.iterdir() if m.is_file()}
                if files_dir.is_dir()
                else set()
            )
            unexpected = present - set(required_files)
            if unexpected:
                raise RegistryError(
                    f"verify_candidate: unexpected candidate file(s) "
                    f"{sorted(unexpected)} for {expected_id}"
                )
            for fname, payload_key in required_files.items():
                ref = payload.get(payload_key)
                if not isinstance(ref, Mapping) or not isinstance(ref.get("sha256"), str):
                    raise RegistryError(
                        f"verify_candidate: payload.{payload_key} is not a blob reference "
                        f"for {expected_id}"
                    )
                member = files_dir / fname
                if not member.is_file():
                    raise RegistryError(
                        f"verify_candidate: missing candidate file {fname} for {expected_id}"
                    )
                raw = member.read_bytes()
                if sha256_bytes(raw) != ref["sha256"]:
                    raise RegistryError(
                        f"verify_candidate: candidate file {fname} does not match "
                        f"payload.{payload_key} for {expected_id}"
                    )
                blob_path = self.blobs_dir / ref["sha256"]
                if not blob_path.is_file() or blob_path.read_bytes() != raw:
                    raise RegistryError(
                        f"verify_candidate: file {fname} is not backed by an intact blob"
                    )
        elif files_dir.is_dir():
            for member in sorted(files_dir.iterdir()):
                if not member.is_file():
                    continue
                raw = member.read_bytes()
                blob_path = self.blobs_dir / sha256_bytes(raw)
                if not blob_path.is_file() or blob_path.read_bytes() != raw:
                    raise RegistryError(
                        f"verify_candidate: file {member.name} is not backed by an intact blob"
                    )

    # -- evidence -----------------------------------------------------------

    def evidence_path(self, evidence_id: str) -> Path:
        return self.evidence_dir / f"{with_sha256_prefix(evidence_id)}.json"

    def write_evidence(
        self,
        kind: Literal["offline_eval", "shadow", "canary"],
        body: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Content-address an evaluation evidence record under ``evidence/<id>.json``.

        The record is ``schema_version`` + ``kind`` + this registry's ``environment`` +
        the caller's ``body`` (its ``backend`` field is preserved, so CI mock evidence is
        permanently marked mock). ``evidence_id = "sha256:" + sha256(canonical(record
        without evidence_id))``, so a given result has a stable id and any later edit
        changes it. Writing an identical record is idempotent.

        Evidence is safety-bearing, so it is rejected fail-closed unless it is COMPLETE:
        ``backend`` must be exactly ``mock``/``real`` (a missing/None/other backend cannot
        be written, so a later demo promotion can trust the field), and ``passed`` must be
        a genuine bool for EVERY kind (a missing/None/0/"true" ``passed`` can never be
        recorded and then silently satisfy a promotion gate).
        """
        if kind not in _EVIDENCE_KINDS:
            raise RegistryError(f"write_evidence: kind must be one of {_EVIDENCE_KINDS}")
        record: dict[str, Any] = {str(k): _jsonable(v) for k, v in body.items()}
        record.pop("evidence_id", None)
        record["schema_version"] = SCHEMA_VERSION
        record["kind"] = kind
        record["registry_environment"] = self.environment
        backend = record.get("backend")
        if backend not in ("mock", "real"):
            raise RegistryError(
                f"write_evidence: backend must be 'mock' or 'real', got {backend!r}"
            )
        # bool is a subclass of int; ``isinstance(x, bool)`` rejects 0/1/None/"true" so a
        # malformed pass flag can never be promoted on.
        if not isinstance(record.get("passed"), bool):
            raise RegistryError(
                f"write_evidence: passed must be a bool, got {record.get('passed')!r}"
            )

        digest = sha256_bytes(canonical_json(record))
        evidence_id = f"sha256:{digest}"
        record["evidence_id"] = evidence_id
        path = self.evidence_path(evidence_id)
        if path.exists():
            return self.load_evidence(evidence_id)
        atomic_write_bytes(path, dumps_bytes(record), read_only=True)
        return record

    def load_evidence(self, evidence_id: str) -> Mapping[str, Any]:
        """Load an evidence record, re-deriving its id to detect tampering (fail-closed)."""
        requested = str(with_sha256_prefix(evidence_id))
        path = self.evidence_path(evidence_id)
        if not path.is_file():
            raise RegistryError(f"load_evidence: unknown evidence {evidence_id}")
        try:
            record = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryError(
                f"load_evidence: evidence {evidence_id} is unreadable ({exc})"
            ) from exc
        if not isinstance(record, dict):
            raise RegistryError(f"load_evidence: record {evidence_id} is not an object")
        body = {k: v for k, v in record.items() if k != "evidence_id"}
        recomputed = f"sha256:{sha256_bytes(canonical_json(body))}"
        if record.get("evidence_id") != recomputed:
            raise RegistryError(f"load_evidence: tampered evidence {evidence_id}")
        # Bind the record to the filename it was requested under: a record copied to a
        # different evidence path (its stored id still self-consistent) must NOT load as
        # the requested id, or an alias could smuggle one result in under another's name.
        if recomputed != requested:
            raise RegistryError(
                f"load_evidence: evidence id mismatch (requested {requested}, "
                f"content is {recomputed})"
            )
        return record

    # -- release-controller path helpers -----------------------------------

    def channel_path(self, channel: str) -> Path:
        return self.channels_dir / channel


def _rmtree_quiet(path: Path) -> None:
    """Best-effort recursive delete of a staging directory (read-only files included)."""
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        try:
            if child.is_dir() and not child.is_symlink():
                child.rmdir()
            else:
                child.chmod(0o600)
                child.unlink()
        except OSError:
            pass
    try:
        path.rmdir()
    except OSError:
        pass
