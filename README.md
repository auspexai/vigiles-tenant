# Vigiles — AuspexAI's first tenant

Vigiles is the first research tenant on the [AuspexAI](https://github.com/auspexai) volunteer compute network. It is an AuspexAI-native clean rebuild for multi-agent LLM behavioral drift research, designed against the [Tenant SDK](https://github.com/auspexai/tenant-sdk) contract from day one.

Vigiles is a sibling project to [Sentinel](https://github.com/jasongagne-git/sentinel), sharing methodological DNA (behavioral drift research) but maintaining its own codebase and roadmap. Sentinel continues to evolve as an independent research program; Vigiles is not a fork or adaptation of Sentinel code.

## Status

**LIVE — the certified, pre-registered starter on the open-beta network (2026-07).** The first experiment is **D6** — a deterministic behavioral-drift probe run against a worker-served local LLM; the minimal package + adaptive driver are in this repo, and the live run needs a worker with inference serving enabled (`[inference] backend = "ollama"`) holding the declared model.

Two findings shaped the current design. The keystone run (`exp-_LtpfHNh`, 2026-06-20) proved two workers *can* produce byte-identical output at temperature 0 — but the first real fleet run (C15, 2026-06-28) proved they reliably **don't** across Ollama versions on a bring-your-own fleet. Corroboration therefore moved to **`within_cell_tolerance@2`**: replicas agree when their declared features fall within each feature's *calibrated comparison envelope* (`type_token_ratio rel ≤ 0.02`, `top_tokens` jaccard ≥ 0.9 — Phase-0-calibrated, locked by the starter's certificate); a version-skewed worker is a recorded **outlier**, never a silent consensus-blocker, and genuine divergence is recorded as data. The starter profile also ships a complete **pre-registered design** — hypothesis, analysis, and stopping rule are Rekor-anchored at submit, so every bundle proves `design ≺ data`.

## Layout

- **`pkg/`** — the tenant package staged on the worker. `executor.py` runs one drift probe per work unit against the worker-served model through the AuspexAI inference broker (W-S, §9 #43) and reduces the response to a sha256 anchor + light lexical features (no raw text — Research Ethics §7 containment). It imports only the vendored `lite.py` (the SDK's stdlib-only `LiteHarness` + `InferenceClient`), so it runs under the worker sandbox's system Python with zero installs.
- **`driver/drift_driver.py`** — the adaptive `run_until` driver. Re-runs a fixed probe panel each round at temperature 0 with a pinned seed and folds the per-probe consensus hashes into a `Counter`; converges once every probe's consensus response holds stable across consecutive rounds (a changed hash for a fixed probe+seed is the drift signal).
- **`experiment.toml`** — the whole build. `[experiment]`/`[executor]`/`[reducer]` feed `auspexai-tenant experiment build pkg/` (SDK-generic; no per-tenant `build.py`), which validates the manifest and computes `executor.package_sha256` over the package files. `[driver]` feeds `experiment run`. It also declares the **`[feature_schema]`, `[benchmark]`** (every emitted feature's meaning, §7-safe bounds, and its consensus `comparison` envelope — the D16.1 self-describing standard), named **`[profiles.*]`** override-sets (`starter` / `research` / `calibration` / `contrast_model`), and the **`[profiles.starter.pre_registration]`** / **`[profiles.research.pre_registration]`** blocks — the pre-registered designs (the envelope is *referenced* from the feature schema, never re-declared). This file is the reference declaration a new tenant copies.
- **`tests/`** — offline executor (vs a fake broker) + driver-convergence tests; no live coordinator or model.

### Running D6

Knobs live in [`experiment.toml`](experiment.toml): `[experiment]`/`[executor]`/`[reducer]` feed the build, `[driver]` feeds the run. Requires `auspexai-tenant>=0.6.51` (the documented `--detach` concurrent flow); the `capture_*` profiles (D20 raw-content capture) require `>=0.6.58`, which harvests the buffered raw live during the run. The legacy `VIGILES_*` env vars still override the *driver* knobs (see [Long-horizon runs](#long-horizon-runs)).

The whole lifecycle is one command (build → submit → await approval → drive):

```sh
auspexai-tenant experiment launch --profile starter
```

`--profile starter` selects the certified, pre-registered configuration; Ctrl-C aborts the run cleanly (server-side too — pass `--resumable` to instead leave it running and resume with `experiment run latest`). Or step-by-step:

```sh
# 1. build pkg/manifest.json from experiment.toml — the label gets a unique
#    suffix stamped on, so re-building then submitting never 409s
auspexai-tenant experiment build pkg/ --profile starter
# 2. one step: sign + upload the package + create the experiment
#    (workers AUTO-FETCH + verify the package, #40a — no staging)
auspexai-tenant experiment submit pkg/ --key <vigiles_key>
# 3. certified runs auto-clear; then drive it — 'latest' resolves the
#    experiment you just submitted, --driver/--journal default from [driver]
auspexai-tenant experiment run latest --key <vigiles_key> --doorbell
```

Generation is deterministic per unit: the worker's inference broker pins `temperature=0` + the seed and authorizes only the manifest's exact model id (seeded sampling — `temperature>0` with a pinned seed, manifest v0.5 — is now honored per-request; sampled replicas use a non-agreement collection mode since they legitimately differ run-to-run). **Consensus, however, is tolerance-based, not byte-based**: identical prompts on different Ollama versions can produce byte-different output, so replicas agree when their declared features fall within the calibrated `comparison` envelope; the attested consensus value is a deterministic *representative* of the agreeing set, and outliers are recorded in the divergence index. The manifest declares the model with `local_weights_required`, which routes units only to workers that hold + serve it.

If your analysis genuinely changes after you've seen data, declare it — append-only and signed, never an edit of the pre-registration:

```sh
auspexai-tenant experiment deviate latest --what "…" --why "…"
```

#### Long-horizon runs

By default the driver is the D6 proof loop: it stops at first stability (`STABLE_ROUNDS` rounds with no new `(probe, hash)` pair) or at 50 rounds. A real longitudinal study keeps observing *past* stability to catch drift — that is **`--profile research`** (8 h duration mode, 5-minute cadence, and its own pre-registered drift *hypothesis*, in contrast to the starter's descriptive baseline). The legacy env knobs parameterize the same driver — all default to D6 behavior when unset:

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

#### Concurrent campaigns (multiple models at once)

`experiment launch` **drives for the whole run**, so several experiments running at once need several persistent drivers — running them one after another in a single terminal drives only the first and leaves the rest **approved with no work units** (driverless). Pass **`--detach`** (`auspexai-tenant` ≥ 0.6.51 — earlier detached builds drove the wrong experiment when several launched at once): each launch runs its driver as a background process that survives the terminal (no `tmux`/`nohup`), and you manage them with `experiment ps` / `experiment stop`:

```sh
auspexai-tenant experiment launch --profile overnight10_mistral --detach
auspexai-tenant experiment launch --profile overnight10_llama   --detach
auspexai-tenant experiment launch --profile overnight10_qwen25  --detach
auspexai-tenant experiment ps                       # confirm all three are driving
auspexai-tenant experiment stop <run-id|exp-id>     # stop one (or --all)
```

Each waits for your maintainer approval, then drives; logs are at `~/.local/share/auspexai-tenant/drivers/<run-id>/driver.log`. A `stopped` row in `ps` whose experiment isn't finished is a driver that died — resume it with `experiment run <exp-id> --detach`. This is the standard for multi-model panels — don't hand-run `launch` in a loop.

## Scope

This repository holds:

- The Vigiles tenant package — executor and driver consuming the AuspexAI [Tenant SDK](https://github.com/auspexai/tenant-sdk)
- Experiment manifests and result schemas for behavioral drift research
- Containment plan implementations per the [Research Ethics Policy](https://github.com/auspexai/.github/blob/main/RESEARCH_ETHICS_POLICY.md) §7

## Releases

Tagged versions are cut as **signed source snapshots** — a `v*` tag triggers the [release workflow](.github/workflows/release.yml). Vigiles is the curated reference tenant, not a wheel, so a release is a `git archive` of the tracked tree, Sigstore-signed (keyless OIDC, like the worker and SDK). This gives the no-code starter on-ramp a stable, verifiable point to pin. To cut one: bump `[project].version` in `pyproject.toml`, then push a matching `vX.Y.Z` tag from green `main` (the workflow guards `tag == pyproject version`). Each release's notes carry the `cosign verify-blob` command; the signing identity is on the [authorized signers](https://github.com/auspexai/.github/blob/main/security/AUTHORIZED_SIGNERS.md) roster.

## Research ethics

Vigiles is the worked example in the AuspexAI Research Ethics Policy (§7). Classification: **medium dual-use risk**. The research subject is behavioral drift — including drift toward harmful behavior. Harmful outputs are research evidence with documented containment, not deliverables. Public reporting uses redacted excerpts, aggregated metrics, and synthetic illustrations; raw transcripts are not publicly released by default.

## License

[Apache-2.0](LICENSE) — per the AuspexAI SDK license boundary design, tenant license is the researcher's choice. Apache-2.0 demonstrates that the AGPL-3.0 platform supports non-AGPL tenants.

## Governance & policies

- [Governance](https://github.com/auspexai/.github/blob/main/GOVERNANCE.md)
- [Code of Conduct](https://github.com/auspexai/.github/blob/main/CODE_OF_CONDUCT.md)
- [Contributing](https://github.com/auspexai/.github/blob/main/CONTRIBUTING.md)
- [Research Ethics Policy](https://github.com/auspexai/.github/blob/main/RESEARCH_ETHICS_POLICY.md)
