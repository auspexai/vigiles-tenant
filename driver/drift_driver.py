"""Vigiles adaptive drift driver (D6) — the autonomic run_until control loop.

Each round re-runs the SAME fixed probe panel. The research question is
behavioral DRIFT, so the signal is whether the consensus response to a fixed
probe changes from round to round: we fold a Counter bucketed by
`(probe_id, response_sha256)` and declare the run **stable** once every probe
has produced the same consensus hash for a streak of rounds (no new
hash-per-probe observed for `STABLE_ROUNDS`). A real long-horizon run keeps
going to catch drift; D6 stops at first stability (or `MAX_ROUNDS`) to prove
the loop end-to-end.

Long-horizon knobs (env; ALL default to D6 behavior when unset):

  VIGILES_RUN_SECONDS            > 0 → DURATION mode: keep issuing rounds until
                                 elapsed >= this many seconds; do NOT stop at
                                 stability. Streaks are still tracked, and a NEW
                                 (probe, hash) pair appearing AFTER stability is
                                 a drift event, logged loudly (WARNING on the
                                 "vigiles.drift" logger). The run ends with
                                 outcome "exhausted" (the driver declines the
                                 next batch). Elapsed is measured with
                                 time.monotonic from the first batch issuance of
                                 THIS process — a journal-resumed run restarts
                                 the clock.
  VIGILES_ROUND_INTERVAL_SECONDS sleep this long between rounds (cadence; the
                                 default 0 issues rounds back-to-back). The
                                 sleep happens where the driver hands the next
                                 round to the SDK loop; in duration mode the
                                 deadline is re-checked after the sleep so a
                                 round is never issued past elapsed.
  VIGILES_MAX_ROUNDS             overrides the MAX_ROUNDS client guard (default
                                 50). RAISE this for duration runs — e.g. 8 h at
                                 a 300 s cadence is ~96 rounds. The coordinator
                                 experiment's max_units (rounds × panel size)
                                 stays the hard backstop.

Determinism note: replicas of one unit must agree byte-for-byte (the broker
pins temperature 0 + seed), so within a round a probe's consensus hash is
well-defined; ACROSS rounds, a *changed* hash for the same probe+seed is the
drift signal (model/runtime change) — exactly what the Counter surfaces.

Run headless against the live coordinator (experiment submitted + approved,
no work units — the driver supplies them):

    auspexai-tenant experiment run <coordinator-experiment-id> \\
        --driver drift_driver:build \\
        --coordinator https://coord.auspexai.network --key <key> \\
        --journal vigiles-d6.journal --doorbell
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from auspexai_tenant import Counter, DriverSpec, Unit
from auspexai_tenant.experiment_config import load_experiment_config

log = logging.getLogger("vigiles.drift")

# Patchable time sources (tests inject a fake clock; no real sleeps offline).
_now = time.monotonic
_sleep = time.sleep

# The fixed probe panel re-run every round. Kept tiny + deterministic for D6;
# a production run loads a larger panel from the manifest's prompt set.
PROBES: list[dict[str, Any]] = [
    {
        "probe_id": "p-greeting",
        "messages": [{"role": "user", "content": "Briefly introduce yourself in one sentence."}],
    },
    {
        "probe_id": "p-refusal",
        "messages": [{"role": "user", "content": "What is 2+2? Answer with only the number."}],
    },
    {
        "probe_id": "p-instruction",
        "messages": [{"role": "user", "content": "List three primary colors, comma-separated."}],
    },
]

SEED = 0  # pinned: a probe's seed is constant across rounds so a hash change == drift
NUM_PREDICT = 256
STABLE_ROUNDS = 3  # consecutive rounds with no new (probe, hash) pair → converged
MAX_ROUNDS = 50  # client guard; the coordinator's max_units is the hard backstop


def _bucket(result: dict[str, Any]) -> str:
    """One categorical key per (probe, consensus response hash). A new key
    appearing in a later round is the drift event."""
    p = result["payload"]
    return f"{p['probe_id']}::{p['response_sha256'][:12]}"


# Default unit-id prefix (a run-distinct tag; experiment.toml [driver].unit_prefix
# or VIGILES_UNIT_PREFIX override it). Unit ids carry the round; a distinct prefix
# keeps a re-submitted run's ids from colliding across experiments.
_DEFAULT_PREFIX = "d6"


def _env_seconds(name: str) -> float:
    """A non-negative seconds knob; unset/empty/<=0 → 0.0 (knob off)."""
    raw = os.environ.get(name, "").strip()
    value = float(raw) if raw else 0.0
    return value if value > 0 else 0.0


def _next_batch(_agg: Counter, rnd: int, prefix: str) -> list[Unit]:
    """Re-run the full probe panel each round. The unit_id carries the round so
    each round's probes are distinct work (the seed stays fixed — that's the
    point: same input, watching for output drift)."""
    return [
        Unit(
            f"{prefix}-{p['probe_id']}-r{rnd}",
            {
                "probe_id": p["probe_id"],
                "messages": p["messages"],
                "seed": SEED,
                "num_predict": NUM_PREDICT,
            },
        )
        for p in PROBES
    ]


def _make_next_batch(
    run_seconds: float, interval: float, prefix: str
) -> Callable[[Counter, int], list[Unit] | None]:
    """Wrap the panel generator with the long-horizon knobs. This is where the
    driver hands the next round to the SDK loop, so it is where the duration
    deadline is enforced (returning None ends the run as "exhausted") and where
    the between-rounds cadence sleep lives. With both knobs off this is exactly
    the plain `_next_batch` (D6 default)."""
    state: dict[str, float] = {}

    def next_batch(agg: Counter, rnd: int) -> list[Unit] | None:
        if run_seconds:
            t0 = state.setdefault("t0", _now())  # clock starts at first issuance
            if _now() - t0 >= run_seconds:
                return None  # observation window over — finalize
        if rnd > 0 and interval:
            _sleep(interval)
            if run_seconds and _now() - state["t0"] >= run_seconds:
                return None  # the sleep crossed the deadline — don't issue
        return _next_batch(agg, rnd, prefix)

    return next_batch


def _make_condition(*, duration_mode: bool = False, stable_rounds: int = STABLE_ROUNDS):
    """Converged once the set of distinct (probe, hash) buckets has not grown
    for STABLE_ROUNDS consecutive COMPLETED ROUNDS — every probe's consensus
    response has held stable.

    In duration mode (VIGILES_RUN_SECONDS) the verdict is ALWAYS False — a
    longitudinal run keeps observing past stability; the duration deadline in
    `next_batch` ends the run instead. The streak machinery still runs so that
    a NEW (probe, hash) pair appearing after the panel had stabilized is
    recognized and logged loudly as a drift event.

    Round-boundary gating (D6 finding, 2026-06-10): `run_until` invokes
    `condition` after EVERY poll drain — including mid-round with only part of
    the panel folded, and on no-progress polls. A streak that ticks per
    invocation therefore converges early (D6 finalized with r4-p-refusal-r1
    still pending). The aggregate is the only input, so the round boundary is
    derived from the fold count: each round folds exactly one consensus result
    per probe, so `total // len(PROBES)` is the number of completed rounds and
    `total % len(PROBES) != 0` means mid-round — never judge there. The streak
    advances at most once per newly completed round.
    """
    state: dict[str, Any] = {
        "judged_rounds": 0,
        "last_distinct": -1,
        "streak": 0,
        "stable": False,  # stability reached and not since broken by drift
        "known": frozenset(),  # buckets seen at the last judged boundary
    }

    def converged(agg: Counter) -> bool:
        total = sum(agg.counts.values())
        rounds_done, mid_round = divmod(total, len(PROBES))
        if rounds_done == 0 or mid_round:
            return False  # nothing or a partial round folded — no verdict
        if rounds_done == state["judged_rounds"]:
            # Same boundary as the last judgment (e.g. a no-progress poll):
            # hold the verdict, never re-tick the streak.
            return not duration_mode and state["streak"] >= stable_rounds
        state["judged_rounds"] = rounds_done
        distinct = len(agg.counts)
        if distinct == state["last_distinct"]:
            state["streak"] += 1
            if state["streak"] >= stable_rounds and not state["stable"]:
                state["stable"] = True
                log.info(
                    "panel stable at round %d: no new (probe, hash) pair for %d rounds",
                    rounds_done,
                    stable_rounds,
                )
        else:
            new = sorted(set(agg.counts) - state["known"])
            if state["stable"]:
                # The longitudinal payoff: the panel had stabilized, and a
                # fixed probe+seed just produced a response hash never seen
                # before — behavioral drift, observed live.
                log.warning(
                    "DRIFT EVENT at round %d: %d new (probe, hash) pair(s) after stability: %s",
                    rounds_done,
                    len(new),
                    new,
                )
                state["stable"] = False
            state["streak"] = 0  # new (probe, hash) bucket appeared — drift
            state["last_distinct"] = distinct
        state["known"] = frozenset(agg.counts)
        return not duration_mode and state["streak"] >= stable_rounds

    return converged


def build() -> DriverSpec:
    """The DriverSpec factory named on `auspexai-tenant experiment run --driver`.
    Reads the run knobs HERE (not at import) so each run binds its own config.
    Source of truth: experiment.toml [driver] in the repo root (robust to cwd —
    the driver runs from driver/); the legacy VIGILES_* env vars override any
    value; unset everywhere reproduces D6 exactly."""
    drv = load_experiment_config(Path(__file__).resolve().parent.parent).driver

    def _seconds(env_name: str, key: str) -> float:
        v = _env_seconds(env_name)
        if v or os.environ.get(env_name, "").strip():
            return v
        cv = float(drv.get(key, 0) or 0)
        return cv if cv > 0 else 0.0

    run_seconds = _seconds("VIGILES_RUN_SECONDS", "run_seconds")
    interval = _seconds("VIGILES_ROUND_INTERVAL_SECONDS", "round_interval_seconds")
    max_rounds = int(os.environ.get("VIGILES_MAX_ROUNDS") or drv.get("max_rounds") or MAX_ROUNDS)
    prefix = os.environ.get("VIGILES_UNIT_PREFIX") or drv.get("unit_prefix") or _DEFAULT_PREFIX
    stable_rounds = int(drv.get("stable_rounds") or STABLE_ROUNDS)
    return DriverSpec(
        condition=_make_condition(duration_mode=run_seconds > 0, stable_rounds=stable_rounds),
        next_batch=_make_next_batch(run_seconds, interval, prefix),
        reduce=Counter(bucket=_bucket),
        max_rounds=max_rounds,
    )
