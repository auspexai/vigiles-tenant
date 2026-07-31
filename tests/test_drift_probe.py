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

from auspexai_tenant import Counter  # noqa: E402

import drift_driver  # noqa: E402
import executor  # noqa: E402  (vendored pkg/executor.py)


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


def test_tokenize_trims_surrounding_punctuation():
    """C7: identical text must tokenize identically regardless of where a comma
    attaches across serving versions — the proof-run false-divergence where
    "blue, red, yellow" diverged only because the comma moved. Interior marks are
    preserved; pure-punctuation tokens dropped; Unicode punctuation handled too."""
    # the exact proof-run case: same response, comma on a different word → identical now
    assert executor._tokenize("blue, red, yellow") == ["blue", "red", "yellow"]
    assert executor._tokenize("blue ,red, yellow,") == ["blue", "red", "yellow"]
    # interior punctuation preserved (apostrophes, interior dots)
    assert executor._tokenize("I'm Gemma.") == ["i'm", "gemma"]
    # Unicode punctuation (curly quotes, em-dash, ellipsis) trimmed, not kept
    assert executor._tokenize("“hello” — world…") == ["hello", "world"]
    # a pure-punctuation token drops out entirely
    assert executor._tokenize("ok ... !!") == ["ok"]


STABLE_PANEL = [
    ("p-greeting", "a" * 64),
    ("p-refusal", "b" * 64),
    ("p-instruction", "c" * 64),
]


def test_driver_converges_on_hash_stability():
    """Observe-all model: the raw run_until path calls `condition` exactly ONCE per
    round (a clean boundary — never mid-round). Convergence lands after STABLE_ROUNDS
    consecutive rounds with no NEW (probe, response) bucket: round 0 introduces the
    buckets, rounds 1..STABLE_ROUNDS add none → converged on the last."""
    cond = drift_driver.build(_cfg()).condition
    agg = Counter(bucket=drift_driver._bucket)
    verdicts = []
    for _round in range(drift_driver.STABLE_ROUNDS + 1):
        for p, h in STABLE_PANEL:
            agg.fold(_consensus(p, h))  # re-observe the same panel — no new bucket after round 0
        verdicts.append(cond(agg))  # one call per round boundary
    assert verdicts == [False] * drift_driver.STABLE_ROUNDS + [True]


def test_stably_divergent_probe_converges():
    """Observe-all (C7 Inc 3 tail): a probe that yields TWO responses every round —
    a STABLE divergence (an Ollama-version split, or a worker that collapses to empty
    output on some replicas) — is a valid stable state, NOT a consensus-blocker. Once
    the observed set stops growing the panel converges, exactly like a single-response
    probe. (This is the case that hung the driver under the old consensus model.)"""
    cond = drift_driver.build(_cfg()).condition
    agg = Counter(bucket=drift_driver._bucket)
    panel = [
        ("p-greeting", "a" * 64),  # greeting diverges across two workers...
        ("p-greeting", "z" * 64),  # ...both responses observed every round
        ("p-refusal", "b" * 64),
        ("p-instruction", "c" * 64),
    ]
    verdicts = []
    for _round in range(drift_driver.STABLE_ROUNDS + 1):
        for p, h in panel:
            agg.fold(_consensus(p, h))
        verdicts.append(cond(agg))
    assert verdicts == [False] * drift_driver.STABLE_ROUNDS + [True]


def test_repeated_observations_of_same_response_dont_perturb_convergence():
    """Observe-all folds EVERY replica: multiple agreeing replicas of the same
    (probe, response) re-increment a bucket's count but add no DISTINCT bucket, so
    they don't perturb the streak — convergence depends only on the distinct
    observed-response SET. (This is also why double-folding on crash-resume is
    harmless: it can only re-increment an existing bucket.)"""
    cond = drift_driver.build(_cfg()).condition
    agg = Counter(bucket=drift_driver._bucket)
    verdicts = []
    for _round in range(drift_driver.STABLE_ROUNDS + 1):
        for p, h in STABLE_PANEL:
            agg.fold(_consensus(p, h))  # replica 1
            agg.fold(_consensus(p, h))  # replica 2 (agrees) — same bucket
            agg.fold(_consensus(p, h))  # a re-fold (e.g. resume) — still one distinct bucket
        verdicts.append(cond(agg))
    assert verdicts == [False] * drift_driver.STABLE_ROUNDS + [True]


def test_driver_resets_streak_on_drift():
    spec = drift_driver.build(_cfg())
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
    units = drift_driver._next_batch(Counter(), 4, "d6", drift_driver.SEED, "fixed")
    assert len(units) == len(drift_driver.PROBES)
    assert {u.unit_id for u in units} == {f"d6-{p['probe_id']}-r4" for p in drift_driver.PROBES}
    assert all(u.payload["seed"] == drift_driver.SEED for u in units)


def test_next_batch_per_round_seed_stream_varies_by_round():
    # seed_policy="per_round" → a declared, deterministic seed-STREAM: each round's seed
    # is base + round (still individually pinned), so the sampler's own range surfaces as
    # dispersion instead of being pinned out (diversity_seed_stream_design.md §3).
    r2 = drift_driver._next_batch(Counter(), 2, "divqs", 100, "per_round")
    r5 = drift_driver._next_batch(Counter(), 5, "divqs", 100, "per_round")
    assert all(u.payload["seed"] == 102 for u in r2)  # base 100 + round 2
    assert all(u.payload["seed"] == 105 for u in r5)  # base 100 + round 5


def test_build_wires_per_round_seed_policy_from_determinism(monkeypatch):
    from auspexai_tenant.experiment_config import ExperimentConfig

    monkeypatch.delenv("VIGILES_SEED_POLICY", raising=False)
    cfg = ExperimentConfig(
        driver={"baseline_rounds": 0, "unit_prefix": "divqs", "max_rounds": 40},
        raw={"determinism": {"temperature": 0.8, "seed": 0, "seed_policy": "per_round"}},
    )
    spec = drift_driver.build(cfg)
    r3 = spec.next_batch(Counter(bucket=drift_driver._bucket), 3)
    assert all(u.payload["seed"] == 3 for u in r3)  # base 0 + round 3 → the stream is live


def _cfg(**driver):
    """A resolved ExperimentConfig the CLI would hand to build(cfg). Driver-logic
    tests pass an explicit (or empty) [driver] table; env vars still override.

    Default `baseline_rounds=0` so the stability/duration/knob suites exercise the
    convergence logic in ISOLATION (legacy D6 semantics unchanged). The production
    default (BASELINE_ROUNDS=5) and the baseline phase itself have dedicated tests
    below — pass an explicit `baseline_rounds=` to opt into it."""
    from auspexai_tenant.experiment_config import ExperimentConfig

    driver.setdefault("baseline_rounds", 0)
    return ExperimentConfig(driver=dict(driver))


def test_build_binds_knobs_from_driver_config(monkeypatch):
    """build(cfg) reads the [driver] knobs (max_rounds, unit_prefix, …) from the
    passed config when the env vars are unset."""
    for var in (
        "VIGILES_RUN_SECONDS",
        "VIGILES_ROUND_INTERVAL_SECONDS",
        "VIGILES_MAX_ROUNDS",
        "VIGILES_UNIT_PREFIX",
    ):
        monkeypatch.delenv(var, raising=False)
    cfg = _cfg(max_rounds=7, unit_prefix="r9")
    spec = drift_driver.build(cfg)
    assert spec.max_rounds == 7  # [driver].max_rounds
    units = spec.next_batch(Counter(bucket=drift_driver._bucket), 0)  # rnd 0 → no cadence sleep
    assert all(u.unit_id.startswith("r9-") for u in units)  # [driver].unit_prefix

    # env still overrides the config.
    monkeypatch.setenv("VIGILES_MAX_ROUNDS", "3")
    assert drift_driver.build(cfg).max_rounds == 3


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
    spec = drift_driver.build(_cfg())
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

    # A D6 (stability) run converges after round-0 warmup + STABLE_ROUNDS = 4 rounds
    # (baseline disabled here); duration mode ran the clock out: 100 s @ 10 s = 10.
    assert rounds == 10
    assert rounds > drift_driver.STABLE_ROUNDS + 1


def test_round_interval_sleeps_between_rounds(monkeypatch):
    """VIGILES_ROUND_INTERVAL_SECONDS: cadence sleep where the driver hands the
    next round to the SDK loop — never before round 0, once per later round.
    Independent of duration mode (here: plain D6 stability mode)."""
    clock = _patch_clock(monkeypatch)
    monkeypatch.setenv("VIGILES_ROUND_INTERVAL_SECONDS", "300")
    spec = drift_driver.build(_cfg())
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
    spec = drift_driver.build(_cfg())
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
    spec = drift_driver.build(_cfg())
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
    spec2 = drift_driver.build(_cfg())
    agg2 = Counter(bucket=drift_driver._bucket)
    with caplog.at_level(logging.WARNING, logger="vigiles.drift"):
        for panel in (STABLE_PANEL, [("p-greeting", "f" * 64), *STABLE_PANEL[1:]]):
            for p, h in panel:
                agg2.fold(_consensus(p, h))
            assert spec2.condition(agg2) is False
    assert not [r for r in caplog.records if "DRIFT EVENT" in r.getMessage()]


def test_max_rounds_env_override(monkeypatch):
    monkeypatch.setenv("VIGILES_MAX_ROUNDS", "120")
    assert drift_driver.build(_cfg()).max_rounds == 120


def test_defaults_unchanged_when_env_unset(monkeypatch):
    """All knobs unset → exact D6 behavior: max_rounds 50, no sleeps, no clock
    reads, stability convergence (the suites above run under the same env)."""
    for var in ("VIGILES_RUN_SECONDS", "VIGILES_ROUND_INTERVAL_SECONDS", "VIGILES_MAX_ROUNDS"):
        monkeypatch.delenv(var, raising=False)

    def _boom(*_a, **_k):
        raise AssertionError("default mode must not touch the clock")

    monkeypatch.setattr(drift_driver, "_sleep", _boom)
    monkeypatch.setattr(drift_driver, "_now", _boom)
    spec = drift_driver.build(_cfg())  # _cfg default disables the baseline → pure D6
    assert spec.max_rounds == drift_driver.MAX_ROUNDS == 50
    agg = Counter(bucket=drift_driver._bucket)
    for rnd in range(2):  # rnd > 0 exercises the (absent) cadence branch
        units = spec.next_batch(agg, rnd)
        assert len(units) == len(drift_driver.PROBES)
    # Stability convergence verdict is unchanged (baseline disabled → streak only).
    verdicts = []
    for _ in range(1 + drift_driver.STABLE_ROUNDS):
        for p, h in STABLE_PANEL:
            agg.fold(_consensus(p, h))
        verdicts.append(spec.condition(agg))
    assert verdicts == [False, False, False, True]


# ---- self-baseline phase (baseline_rounds, K) — the first K rounds calibrate ----
# ---- this model's OWN reference, then drift is measured from it (§3.1)        ----


def _drive(spec, panels):
    """Replicate the raw run_until loop shape: `condition` is checked ONCE per round
    at the top (BEFORE that round's `next_batch`, which records `phase['round']`),
    then the round is folded. So the r-th verdict sees exactly r rounds of data —
    identical to production. Returns (verdicts, aggregate); stops at first True,
    like the loop breaking before it issues the converged round."""
    agg = Counter(bucket=drift_driver._bucket)
    verdicts = []
    for rnd, panel in enumerate(panels):
        v = spec.condition(agg)  # top-of-loop check: sees rounds 0..rnd-1
        verdicts.append(v)
        if v:
            return verdicts, agg  # converged — the loop would break before issuing
        spec.next_batch(agg, rnd)  # records phase['round']=rnd
        for p, h in panel:
            agg.fold(_consensus(p, h))
    verdicts.append(spec.condition(agg))
    return verdicts, agg


def test_baseline_defers_convergence_until_the_full_window():
    """The floor: with K=6, a perfectly stable panel that WOULD converge on stability
    at round 4 (streak >= STABLE_ROUNDS) is held back — the run spends the whole
    baseline window and converges exactly at round K, freezing a K-round reference."""
    verdicts, _ = _drive(drift_driver.build(_cfg(baseline_rounds=6)), [STABLE_PANEL] * 10)
    assert verdicts.index(True) == 6  # deferred to the full baseline, not stability at 4
    assert all(v is False for v in verdicts[:6])
    # sanity: with the baseline disabled the same panel converges at 4 (streak only).
    legacy, _ = _drive(drift_driver.build(_cfg(baseline_rounds=0)), [STABLE_PANEL] * 10)
    assert legacy.index(True) == 4


def test_baseline_rounds_default_is_five(monkeypatch):
    """An untouched config (no [driver].baseline_rounds, env unset) uses the shipped
    default BASELINE_ROUNDS=5 — Vigiles spends a self-baseline by DEFAULT, not opt-in."""
    from auspexai_tenant.experiment_config import ExperimentConfig

    monkeypatch.delenv("VIGILES_BASELINE_ROUNDS", raising=False)
    assert drift_driver.BASELINE_ROUNDS == 5
    verdicts, _ = _drive(drift_driver.build(ExperimentConfig(driver={})), [STABLE_PANEL] * 9)
    assert verdicts.index(True) == 5


def test_baseline_rounds_env_overrides_and_clamps_to_max_rounds(monkeypatch):
    # env beats [driver].baseline_rounds=6; both exceed the streak point (4) so the
    # baseline is the binding constraint and convergence lands exactly at K.
    monkeypatch.setenv("VIGILES_BASELINE_ROUNDS", "8")
    verdicts, _ = _drive(drift_driver.build(_cfg(baseline_rounds=6)), [STABLE_PANEL] * 12)
    assert verdicts.index(True) == 8  # env (8), not config (6)
    monkeypatch.delenv("VIGILES_BASELINE_ROUNDS")
    # a baseline longer than the run is clamped to max_rounds — without the clamp,
    # baseline=100 would defer past the run and it could never complete the window.
    spec = drift_driver.build(_cfg(baseline_rounds=100, max_rounds=7))
    verdicts, _ = _drive(spec, [STABLE_PANEL] * 12)
    assert verdicts.index(True) == 7  # clamped to max_rounds (else it never converges)


def test_monitoring_flags_drift_from_the_frozen_baseline(monkeypatch, caplog):
    """After the K-round baseline freezes, a fixed probe producing a response ABSENT
    from that baseline is drift FROM this model's own normal — logged as a DRIFT EVENT
    'beyond the baseline'. Duration mode keeps monitoring past the baseline."""
    _patch_clock(monkeypatch)
    monkeypatch.setenv("VIGILES_RUN_SECONDS", "999999")  # never converge — keep monitoring
    spec = drift_driver.build(_cfg(baseline_rounds=3))
    agg = Counter(bucket=drift_driver._bucket)
    caplog.set_level(logging.INFO, logger="vigiles.drift")

    for rnd in range(3):  # rounds 0..2 establish the reference (3 stable buckets)
        assert spec.condition(agg) is False  # in-baseline: never converges
        spec.next_batch(agg, rnd)
        for p, h in STABLE_PANEL:
            agg.fold(_consensus(p, h))
    assert spec.condition(agg) is False  # round-3 top check: freezes the baseline
    assert any("baseline established: 3" in r.getMessage() for r in caplog.records)

    spec.next_batch(agg, 3)  # first monitoring round
    caplog.clear()
    agg.fold(_consensus("p-greeting", "f" * 64))  # a NEW response for a fixed probe
    for p, h in STABLE_PANEL[1:]:
        agg.fold(_consensus(p, h))
    assert spec.condition(agg) is False  # duration mode → no stop, but the event fires
    events = [r.getMessage() for r in caplog.records if "DRIFT EVENT" in r.getMessage()]
    assert len(events) == 1
    assert "beyond the baseline" in events[0]
    assert "p-greeting::ffffffffffff" in events[0]


def test_baseline_absorbs_variation_so_it_is_not_flagged_in_monitoring(monkeypatch, caplog):
    """A response first seen DURING the baseline (natural run-to-run variation, not
    just round 0) is part of the reference set — its later reappearance in monitoring
    is NOT drift. This is why the calibration window must be spent, not assumed."""
    _patch_clock(monkeypatch)
    monkeypatch.setenv("VIGILES_RUN_SECONDS", "999999")
    spec = drift_driver.build(_cfg(baseline_rounds=4))
    agg = Counter(bucket=drift_driver._bucket)

    # Baseline rounds 0..3: p-greeting varies between two responses (a, then z) —
    # both belong to this model's normal, so the frozen baseline holds both.
    baseline_panels = [
        STABLE_PANEL,
        [("p-greeting", "z" * 64), *STABLE_PANEL[1:]],
        STABLE_PANEL,
        [("p-greeting", "z" * 64), *STABLE_PANEL[1:]],
    ]
    for rnd, panel in enumerate(baseline_panels):
        assert spec.condition(agg) is False
        spec.next_batch(agg, rnd)
        for p, h in panel:
            agg.fold(_consensus(p, h))
    assert spec.condition(agg) is False  # freeze: baseline now holds greeting a AND z

    caplog.set_level(logging.WARNING, logger="vigiles.drift")
    # Monitoring: greeting flips back to 'a' (in baseline) — must NOT be a drift event.
    spec.next_batch(agg, 4)
    for p, h in STABLE_PANEL:
        agg.fold(_consensus(p, h))
    assert spec.condition(agg) is False
    assert not [r for r in caplog.records if "DRIFT EVENT" in r.getMessage()]


def test_next_batch_logs_baseline_and_monitoring_transitions(caplog):
    spec = drift_driver.build(_cfg(baseline_rounds=3))
    agg = Counter(bucket=drift_driver._bucket)
    caplog.set_level(logging.INFO, logger="vigiles.drift")
    for rnd in range(5):
        spec.next_batch(agg, rnd)
    msgs = "\n".join(r.getMessage() for r in caplog.records)
    assert "baseline phase: establishing" in msgs
    assert "monitoring phase begins (round 3)" in msgs
