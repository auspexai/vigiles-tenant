"""Vigiles D6 drift-probe — executor (vs a fake broker) + driver convergence.

Offline, no live coordinator and no real model: the executor runs against a
unix-socket fake speaking the worker broker protocol; the driver's condition
is exercised with synthetic Counter folds. Proves the unit produces a
deterministic, raw-text-free payload and that the run converges on hash
stability.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pkg"))
sys.path.insert(0, str(ROOT / "driver"))

import executor  # noqa: E402  (vendored pkg/executor.py)
import drift_driver  # noqa: E402
from auspexai_tenant import Counter  # noqa: E402


class _FakeBroker:
    """Unix-socket fake speaking the worker broker wire protocol. Returns a
    fixed response per probe so the executor output is deterministic."""

    def __init__(self, socket_path: Path, content: str, digest: str) -> None:
        self.socket_path = socket_path
        self._content = content
        self._digest = digest
        self.requests: list[dict] = []
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(str(socket_path))
        self._listener.listen(2)
        self._listener.settimeout(0.5)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with conn:
                buf = b""
                while b"\n" not in buf:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                if b"\n" not in buf:
                    continue
                req = json.loads(buf.split(b"\n", 1)[0])
                self.requests.append(req)
                if req["op"] == "info":
                    reply = {
                        "ok": True,
                        "model": "gemma-3-1b-q4",
                        "gguf_sha256": self._digest,
                        "backend_handle": "auspex-gemma-3-1b-q4",
                    }
                else:
                    reply = {
                        "ok": True,
                        "message": {"role": "assistant", "content": self._content},
                        "eval_count": 11,
                        "model": "gemma-3-1b-q4",
                    }
                conn.sendall(json.dumps(reply).encode() + b"\n")

    def close(self) -> None:
        self._stop.set()
        self._listener.close()
        self._thread.join(timeout=2.0)


def test_executor_payload_is_deterministic_and_raw_free(tmp_path, monkeypatch):
    sock = tmp_path / "inference.sock"
    content = "alpha beta alpha gamma"  # tokens=4, unique=3, ttr=0.75
    broker = _FakeBroker(sock, content, "ab" * 32)
    try:
        monkeypatch.setenv("AUSPEXAI_INFERENCE_SOCKET", str(sock))
        monkeypatch.setenv("AUSPEXAI_INFERENCE_MODEL", "gemma-3-1b-q4")
        unit = {
            "schema_version": "0.1",
            "unit_id": "p-greeting-r0",
            "tenant_id": "vigiles",
            "experiment_id": "vigiles-d6",
            "manifest_sha256": "cd" * 32,
            "created_at": "2026-06-09T12:00:00+00:00",
            "payload": {
                "probe_id": "p-greeting",
                "messages": [{"role": "user", "content": "hi"}],
                "seed": 0,
            },
        }
        out = executor.run_one(unit, tmp_path)
    finally:
        broker.close()

    import hashlib

    assert out["schema"] == "vigiles-drift-probe/v0"
    assert out["probe_id"] == "p-greeting"
    # The drift/consensus anchor is the hash, not the text — raw output is NOT
    # in the payload (Research Ethics §7 containment).
    assert out["response_sha256"] == hashlib.sha256(content.encode()).hexdigest()
    assert out["lexical"]["tokens"] == 4
    assert out["lexical"]["unique_tokens"] == 3  # "alpha" repeats
    assert out["lexical"]["type_token_ratio"] == 0.75
    assert out["lexical"]["top_tokens"][0] == ["alpha", 2]  # count desc, token asc
    assert out["model"]["gguf_sha256"] == "ab" * 32

    # Determinism: same input → byte-identical payload (consensus precondition).
    broker2 = _FakeBroker(tmp_path / "s2.sock", content, "ab" * 32)
    try:
        monkeypatch.setenv("AUSPEXAI_INFERENCE_SOCKET", str(tmp_path / "s2.sock"))
        out2 = executor.run_one(unit, tmp_path)
    finally:
        broker2.close()
    assert json.dumps(out, sort_keys=True) == json.dumps(out2, sort_keys=True)


def _consensus(probe_id: str, digest: str) -> dict:
    """A consensus-result shape the Counter bucket reads."""
    return {"payload": {"probe_id": probe_id, "response_sha256": digest}}


STABLE_PANEL = [
    ("p-greeting", "a" * 64),
    ("p-refusal", "b" * 64),
    ("p-instruction", "c" * 64),
]


def test_driver_converges_on_hash_stability():
    """run_until calls `condition` after EVERY drain — mid-round included
    (driver.py poll loop), not once per round. Convergence must land only at
    the round boundary after STABLE_ROUNDS stable completed rounds: baseline
    round 0, stable rounds 1-3 → converged at the end of round 3."""
    spec = drift_driver.build()
    cond = spec.condition
    agg = Counter(bucket=drift_driver._bucket)

    boundary_verdicts = []
    for _round in range(4):
        for p, h in STABLE_PANEL:
            agg.fold(_consensus(p, h))
            mid = cond(agg)  # the real per-drain call pattern
            if sum(agg.counts.values()) % len(drift_driver.PROBES) != 0:
                assert mid is False, "verdict on a partial round"
        boundary_verdicts.append(cond(agg))
    assert boundary_verdicts == [False, False, False, True]


def test_driver_never_converges_mid_round_on_no_progress_polls():
    """D6 regression (2026-06-10): the run finalized with r4-p-refusal-r1 still
    pending — repeated no-progress polls ticked the streak inside one round.
    With a partial round folded, any number of condition calls must stay False."""
    spec = drift_driver.build()
    cond = spec.condition
    agg = Counter(bucket=drift_driver._bucket)

    # Round 0 completes (baseline)...
    for p, h in STABLE_PANEL:
        agg.fold(_consensus(p, h))
    assert cond(agg) is False
    # ...round 1 folds only 2 of 3 probes, then the driver polls in a loop.
    for p, h in STABLE_PANEL[:2]:
        agg.fold(_consensus(p, h))
    for _poll in range(drift_driver.STABLE_ROUNDS * 5):
        assert cond(agg) is False, "streak ticked on a no-progress poll"
    # Completing rounds 1-3 with no new bucket converges exactly on schedule.
    agg.fold(_consensus(*STABLE_PANEL[2]))
    assert cond(agg) is False  # round 1: streak 1
    for _round in (2, 3):
        for p, h in STABLE_PANEL:
            agg.fold(_consensus(p, h))
        verdict = cond(agg)
    assert verdict is True


def test_driver_holds_verdict_when_repolled_at_same_boundary():
    """A completed-round boundary judged twice (poll with no new folds) must
    hold its verdict without advancing the streak."""
    spec = drift_driver.build()
    cond = spec.condition
    agg = Counter(bucket=drift_driver._bucket)
    for p, h in STABLE_PANEL:
        agg.fold(_consensus(p, h))
    # Baseline round judged repeatedly: streak must stay 0, never converge.
    for _ in range(drift_driver.STABLE_ROUNDS * 5):
        assert cond(agg) is False


def test_driver_resets_streak_on_drift():
    spec = drift_driver.build()
    cond = spec.condition
    agg = Counter(bucket=drift_driver._bucket)
    for _ in range(1 + drift_driver.STABLE_ROUNDS - 1):  # baseline + 2 stable rounds
        for p, h in STABLE_PANEL:
            agg.fold(_consensus(p, h))
        assert cond(agg) is False
    # Next round: one probe DRIFTS (new bucket) → streak resets at the boundary...
    agg.fold(_consensus("p-greeting", "f" * 64))
    for p, h in STABLE_PANEL[1:]:
        agg.fold(_consensus(p, h))
    assert cond(agg) is False
    # ...so stability must be re-earned over STABLE_ROUNDS full rounds. The
    # drifted hash is now part of the stable panel (4 buckets, no growth).
    drifted = [("p-greeting", "f" * 64), *STABLE_PANEL[1:]]
    verdicts = []
    for _ in range(drift_driver.STABLE_ROUNDS):
        for p, h in drifted:
            agg.fold(_consensus(p, h))
        verdicts.append(cond(agg))
    assert verdicts == [False, False, True]


def test_next_batch_reruns_full_panel_with_fixed_seed():
    units = drift_driver._next_batch(Counter(), 4)
    assert len(units) == len(drift_driver.PROBES)
    assert {u.unit_id for u in units} == {
        f"{drift_driver.UNIT_PREFIX}-{p['probe_id']}-r4" for p in drift_driver.PROBES
    }
    assert all(u.payload["seed"] == drift_driver.SEED for u in units)
