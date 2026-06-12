"""Vigiles D6 drift-probe — executor (vs a fake broker) + driver convergence.

Offline, no live coordinator and no real model: the executor runs against a
unix-socket fake speaking the worker broker protocol; the driver's condition
is exercised with synthetic Counter folds. Proves the unit produces a
deterministic, raw-text-free payload and that the run converges on hash
stability.
"""

from __future__ import annotations

import json
import logging
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


# ---- long-horizon knobs (VIGILES_RUN_SECONDS / _ROUND_INTERVAL_SECONDS / ----
# ---- _MAX_ROUNDS) — offline, fake clock, no real sleeps                  ----


class _FakeClock:
    """Injectable monotonic clock: `now`/`sleep` patch drift_driver._now/_sleep
    so duration + cadence logic runs offline with zero real waiting."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.t = start
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _patch_clock(monkeypatch) -> _FakeClock:
    clock = _FakeClock()
    monkeypatch.setattr(drift_driver, "_now", clock.now)
    monkeypatch.setattr(drift_driver, "_sleep", clock.sleep)
    return clock


def test_duration_mode_keeps_issuing_past_stability_until_elapsed(monkeypatch):
    """VIGILES_RUN_SECONDS: a longitudinal run does NOT stop at stability — it
    keeps issuing rounds (condition always False) and ends when next_batch
    declines the round after the elapsed window (run_until outcome: exhausted)."""
    clock = _patch_clock(monkeypatch)
    monkeypatch.setenv("VIGILES_RUN_SECONDS", "100")
    spec = drift_driver.build()
    agg = Counter(bucket=drift_driver._bucket)

    rounds = 0
    while True:
        units = spec.next_batch(agg, rounds)
        if not units:
            break
        assert len(units) == len(drift_driver.PROBES)
        for p, h in STABLE_PANEL:
            agg.fold(_consensus(p, h))
        assert spec.condition(agg) is False, "duration mode stopped at stability"
        clock.advance(10.0)  # each round takes 10 s of fake wall-clock
        rounds += 1

    # A D6 (stability) run converges after baseline + STABLE_ROUNDS = 4 rounds;
    # duration mode ran the clock out instead: 100 s at 10 s/round = 10 rounds.
    assert rounds == 10
    assert rounds > drift_driver.STABLE_ROUNDS + 1


def test_round_interval_sleeps_between_rounds(monkeypatch):
    """VIGILES_ROUND_INTERVAL_SECONDS: cadence sleep where the driver hands the
    next round to the SDK loop — never before round 0, once per later round.
    Independent of duration mode (here: plain D6 stability mode)."""
    clock = _patch_clock(monkeypatch)
    monkeypatch.setenv("VIGILES_ROUND_INTERVAL_SECONDS", "300")
    spec = drift_driver.build()
    agg = Counter(bucket=drift_driver._bucket)

    assert spec.next_batch(agg, 0)
    assert clock.sleeps == [], "slept before the first round"
    assert spec.next_batch(agg, 1)
    assert spec.next_batch(agg, 2)
    assert clock.sleeps == [300.0, 300.0]


def test_interval_sleep_crossing_deadline_does_not_issue(monkeypatch):
    """Duration + cadence together: if the between-rounds sleep crosses the
    elapsed deadline, the next round is NOT issued (deadline re-checked after
    the sleep)."""
    clock = _patch_clock(monkeypatch)
    monkeypatch.setenv("VIGILES_RUN_SECONDS", "100")
    monkeypatch.setenv("VIGILES_ROUND_INTERVAL_SECONDS", "60")
    spec = drift_driver.build()
    agg = Counter(bucket=drift_driver._bucket)

    assert spec.next_batch(agg, 0)  # t0 anchored here
    clock.advance(50.0)  # round 0 took 50 s — still inside the window
    assert spec.next_batch(agg, 1) is None  # sleep(60) → elapsed 110 >= 100
    assert clock.sleeps == [60.0]


def test_duration_mode_logs_drift_event_after_stability(monkeypatch, caplog):
    """The longitudinal payoff: streaks are still tracked in duration mode, and
    a NEW (probe, hash) pair appearing AFTER stability is logged loudly."""
    _patch_clock(monkeypatch)
    monkeypatch.setenv("VIGILES_RUN_SECONDS", "999999")
    spec = drift_driver.build()
    cond = spec.condition
    agg = Counter(bucket=drift_driver._bucket)

    # Baseline + STABLE_ROUNDS identical rounds → stability reached, no stop.
    for _ in range(1 + drift_driver.STABLE_ROUNDS):
        for p, h in STABLE_PANEL:
            agg.fold(_consensus(p, h))
        assert cond(agg) is False
    # One probe DRIFTS after stability → loud drift event, still no stop.
    with caplog.at_level(logging.WARNING, logger="vigiles.drift"):
        agg.fold(_consensus("p-greeting", "f" * 64))
        for p, h in STABLE_PANEL[1:]:
            agg.fold(_consensus(p, h))
        assert cond(agg) is False
    events = [r.getMessage() for r in caplog.records if "DRIFT EVENT" in r.getMessage()]
    assert len(events) == 1
    assert "p-greeting::ffffffffffff" in events[0]

    # Pre-stability bucket growth (rounds 0→1 of a fresh run) is NOT an event.
    caplog.clear()
    spec2 = drift_driver.build()
    agg2 = Counter(bucket=drift_driver._bucket)
    with caplog.at_level(logging.WARNING, logger="vigiles.drift"):
        for panel in (STABLE_PANEL, [("p-greeting", "f" * 64), *STABLE_PANEL[1:]]):
            for p, h in panel:
                agg2.fold(_consensus(p, h))
            assert spec2.condition(agg2) is False
    assert not [r for r in caplog.records if "DRIFT EVENT" in r.getMessage()]


def test_max_rounds_env_override(monkeypatch):
    monkeypatch.setenv("VIGILES_MAX_ROUNDS", "120")
    assert drift_driver.build().max_rounds == 120


def test_defaults_unchanged_when_env_unset(monkeypatch):
    """All knobs unset → exact D6 behavior: max_rounds 50, no sleeps, no clock
    reads, stability convergence (the suites above run under the same env)."""
    for var in ("VIGILES_RUN_SECONDS", "VIGILES_ROUND_INTERVAL_SECONDS", "VIGILES_MAX_ROUNDS"):
        monkeypatch.delenv(var, raising=False)

    def _boom(*_a, **_k):
        raise AssertionError("default mode must not touch the clock")

    monkeypatch.setattr(drift_driver, "_sleep", _boom)
    monkeypatch.setattr(drift_driver, "_now", _boom)
    spec = drift_driver.build()
    assert spec.max_rounds == drift_driver.MAX_ROUNDS == 50
    agg = Counter(bucket=drift_driver._bucket)
    for rnd in range(2):  # rnd > 0 exercises the (absent) cadence branch
        units = spec.next_batch(agg, rnd)
        assert len(units) == len(drift_driver.PROBES)
    # Stability convergence verdict is unchanged (baseline + STABLE_ROUNDS).
    verdicts = []
    for _ in range(1 + drift_driver.STABLE_ROUNDS):
        for p, h in STABLE_PANEL:
            agg.fold(_consensus(p, h))
        verdicts.append(spec.condition(agg))
    assert verdicts == [False, False, False, True]
