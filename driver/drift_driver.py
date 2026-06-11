"""Vigiles adaptive drift driver (D6) — the autonomic run_until control loop.

Each round re-runs the SAME fixed probe panel. The research question is
behavioral DRIFT, so the signal is whether the consensus response to a fixed
probe changes from round to round: we fold a Counter bucketed by
`(probe_id, response_sha256)` and declare the run **stable** once every probe
has produced the same consensus hash for a streak of rounds (no new
hash-per-probe observed for `STABLE_ROUNDS`). A real long-horizon run keeps
going to catch drift; D6 stops at first stability (or `MAX_ROUNDS`) to prove
the loop end-to-end.

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

import os
from typing import Any

from auspexai_tenant import Counter, DriverSpec, Unit

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


# Run-unique unit-id prefix. Unit ids MUST NOT collide across experiments:
# the worker's local bookkeeping keys by unit_id (worker bug, tracked), so a
# resubmitted run reusing ids can get its results misattributed to a prior
# experiment's assignments. Until the worker fix lands, every run sets its own
# prefix (e.g. VIGILES_UNIT_PREFIX=r4 → "r4-p-greeting-r0").
UNIT_PREFIX = os.environ.get("VIGILES_UNIT_PREFIX", "d6")


def _next_batch(_agg: Counter, rnd: int) -> list[Unit]:
    """Re-run the full probe panel each round. The unit_id carries the round so
    each round's probes are distinct work (the seed stays fixed — that's the
    point: same input, watching for output drift)."""
    return [
        Unit(
            f"{UNIT_PREFIX}-{p['probe_id']}-r{rnd}",
            {
                "probe_id": p["probe_id"],
                "messages": p["messages"],
                "seed": SEED,
                "num_predict": NUM_PREDICT,
            },
        )
        for p in PROBES
    ]


def _make_condition():
    """Converged once the set of distinct (probe, hash) buckets has not grown
    for STABLE_ROUNDS consecutive COMPLETED ROUNDS — every probe's consensus
    response has held stable.

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
    state = {"judged_rounds": 0, "last_distinct": -1, "streak": 0}

    def converged(agg: Counter) -> bool:
        total = sum(agg.counts.values())
        rounds_done, mid_round = divmod(total, len(PROBES))
        if rounds_done == 0 or mid_round:
            return False  # nothing or a partial round folded — no verdict
        if rounds_done == state["judged_rounds"]:
            # Same boundary as the last judgment (e.g. a no-progress poll):
            # hold the verdict, never re-tick the streak.
            return state["streak"] >= STABLE_ROUNDS
        state["judged_rounds"] = rounds_done
        distinct = len(agg.counts)
        if distinct == state["last_distinct"]:
            state["streak"] += 1
        else:
            state["streak"] = 0  # new (probe, hash) bucket appeared — drift
            state["last_distinct"] = distinct
        return state["streak"] >= STABLE_ROUNDS

    return converged


def build() -> DriverSpec:
    """The DriverSpec factory named on `auspexai-tenant experiment run --driver`."""
    return DriverSpec(
        condition=_make_condition(),
        next_batch=_next_batch,
        reduce=Counter(bucket=_bucket),
        max_rounds=MAX_ROUNDS,
    )
