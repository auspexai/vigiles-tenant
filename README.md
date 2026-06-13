# Vigiles — AuspexAI's first tenant

Vigiles is the first research tenant on the [AuspexAI](https://github.com/auspexai) volunteer compute network. It is an AuspexAI-native clean rebuild for multi-agent LLM behavioral drift research, designed against the [Tenant SDK](https://github.com/auspexai/tenant-sdk) contract from day one.

Vigiles is a sibling project to [Sentinel](https://github.com/jasongagne-git/sentinel), sharing methodological DNA (behavioral drift research) but maintaining its own codebase and roadmap. Sentinel continues to evolve as an independent research program; Vigiles is not a fork or adaptation of Sentinel code.

## Status

**D6 package built (2026-06-09); awaiting the first live run.** Vigiles is sequenced last in Phase 1 (per [Principles & Scope §5.3](https://github.com/auspexai/platform)) so the platform interfaces prove themselves against the synthetic test tenant before bending toward a real tenant's needs. The first experiment is **D6** — a deterministic behavioral-drift probe run against a worker-served local LLM. The minimal package + adaptive driver are in this repo; the live run needs a worker with inference serving enabled (`[inference] backend = "ollama"`) holding the declared model.

## Layout

- **`pkg/`** — the tenant package staged on the worker. `executor.py` runs one drift probe per work unit against the worker-served model through the AuspexAI inference broker (W-S, §9 #43) and reduces the response to a sha256 anchor + light lexical features (no raw text — Research Ethics §7 containment). It imports only the vendored `lite.py` (the SDK's stdlib-only `LiteHarness` + `InferenceClient`), so it runs under the worker sandbox's system Python with zero installs.
- **`driver/drift_driver.py`** — the adaptive `run_until` driver. Re-runs a fixed probe panel each round at temperature 0 with a pinned seed and folds the per-probe consensus hashes into a `Counter`; converges once every probe's consensus response holds stable across consecutive rounds (a changed hash for a fixed probe+seed is the drift signal).
- **`build.py`** — builds + validates the manifest and computes `executor.package_sha256` over the package files. `VIGILES_MODEL_ID=<store-id>` selects the served model.
- **`tests/`** — offline executor (vs a fake broker) + driver-convergence tests; no live coordinator or model.

### Running D6

Knobs live in [`experiment.toml`](experiment.toml) (`[experiment]` + `[driver]`); the legacy `VIGILES_*` env vars still override any value. Requires `auspexai-tenant>=0.5.8`.

```sh
# 1. build pkg/manifest.json from experiment.toml — the label gets a unique
#    suffix stamped on, so re-building then submitting never 409s
python build.py
# 2. one step: sign + upload the package + create the experiment
#    (workers AUTO-FETCH + verify the package, #40a — no staging)
auspexai-tenant experiment submit pkg/ --key <vigiles_key>
# 3. maintainer approves in the console; then drive it — 'latest' resolves the
#    experiment you just submitted, --driver/--journal default from [driver]
cd driver && auspexai-tenant experiment run latest --key <vigiles_key> --doorbell
```

Determinism is consensus-critical: the worker's inference broker pins `temperature=0` + the seed and authorizes only the manifest's exact model id, so replicas of a unit produce byte-identical payloads (the hash-agreement precondition). The manifest declares the model with `local_weights_required`, which routes units only to workers that hold + serve it.

#### Long-horizon runs

By default the driver is the D6 proof loop: it stops at first stability (`STABLE_ROUNDS` rounds with no new `(probe, hash)` pair) or at 50 rounds. A real longitudinal study keeps observing *past* stability to catch drift. Three env knobs parameterize the same driver — all default to D6 behavior when unset:

| Knob | Effect |
| --- | --- |
| `VIGILES_RUN_SECONDS` | `> 0` → duration mode: keep issuing rounds until this much wall-clock has elapsed; do **not** stop at stability. A new `(probe, hash)` pair after stability is a **drift event**, logged loudly on the `vigiles.drift` logger. |
| `VIGILES_ROUND_INTERVAL_SECONDS` | Cadence: sleep this long between rounds (default `0` = back-to-back). |
| `VIGILES_MAX_ROUNDS` | Overrides the 50-round client guard — raise it for duration runs. |

Overnight example — ~8 h at a 5-minute cadence ≈ 96 rounds × 3 probes = 288 units (make sure the experiment's `max_units` on the coordinator covers it; it stays the hard backstop):

```sh
VIGILES_RUN_SECONDS=28800 VIGILES_ROUND_INTERVAL_SECONDS=300 VIGILES_MAX_ROUNDS=120 \
VIGILES_UNIT_PREFIX=<run-prefix> \
auspexai-tenant experiment run <coordinator-experiment-id> \
    --driver drift_driver:build \
    --coordinator https://coord.auspexai.network --key <vigiles_key> \
    --journal vigiles-overnight.journal --doorbell
```

A duration run ends with outcome `exhausted` (the driver declines the round after the window closes — observation complete), not `converged`. The elapsed clock is `time.monotonic` from the first batch this process issues; a journal-resumed run restarts the window.

## Scope

This repository holds:

- The Vigiles tenant package — executor and driver consuming the AuspexAI [Tenant SDK](https://github.com/auspexai/tenant-sdk)
- Experiment manifests and result schemas for behavioral drift research
- Containment plan implementations per the [Research Ethics Policy](https://github.com/auspexai/.github/blob/main/RESEARCH_ETHICS_POLICY.md) §7

## Research ethics

Vigiles is the worked example in the AuspexAI Research Ethics Policy (§7). Classification: **medium dual-use risk**. The research subject is behavioral drift — including drift toward harmful behavior. Harmful outputs are research evidence with documented containment, not deliverables. Public reporting uses redacted excerpts, aggregated metrics, and synthetic illustrations; raw transcripts are not publicly released by default.

## License

[Apache-2.0](LICENSE) — per the AuspexAI SDK license boundary design, tenant license is the researcher's choice. Apache-2.0 demonstrates that the AGPL-3.0 platform supports non-AGPL tenants.

## Governance & policies

- [Governance](https://github.com/auspexai/.github/blob/main/GOVERNANCE.md)
- [Code of Conduct](https://github.com/auspexai/.github/blob/main/CODE_OF_CONDUCT.md)
- [Contributing](https://github.com/auspexai/.github/blob/main/CONTRIBUTING.md)
- [Research Ethics Policy](https://github.com/auspexai/.github/blob/main/RESEARCH_ETHICS_POLICY.md)
