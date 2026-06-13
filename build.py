#!/usr/bin/env python3
"""Build the Vigiles D6 tenant package + manifest.

Produces the signed-ready manifest (with executor.package_sha256 over the
package files = executor.py + the vendored lite.py) and prints the
manifest_sha256 plus the stage / submit / run commands. Run from a venv with
auspexai_tenant installed.

The MODEL_ID must match a model the target worker has in its BYOM store AND
serves (worker `[inference] backend = "ollama"`). Override via env:

    VIGILES_MODEL_ID=<store-model-id> python build.py
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from auspexai_tenant.experiment_config import load_experiment_config, make_unique_label
from auspexai_tenant.manifest import Manifest, compute_package_digest

HERE = Path(__file__).parent
PKG = HERE / "pkg"  # executor.py + lite.py (both digested into package_sha256)

# experiment.toml is the source of truth; the legacy VIGILES_* env vars still
# override any value (so existing one-liners keep working).
_CFG = load_experiment_config(HERE)
_EXP = _CFG.experiment

TENANT_ID = os.environ.get("VIGILES_TENANT_ID") or _EXP.get("tenant_id", "vigiles-lab")
MODEL_ID = os.environ.get("VIGILES_MODEL_ID") or _CFG.model_id or "gemma-3-1b-it-q4"
_BASE_LABEL = os.environ.get("VIGILES_LABEL") or _EXP.get("label", "vigiles-d6-drift")
# Labels are unique FOREVER (aborted runs keep theirs), so stamp a timestamp
# suffix → re-building then `experiment submit` never 409s. VIGILES_EXACT_LABEL=1
# keeps the base label verbatim.
LABEL = (
    _BASE_LABEL if os.environ.get("VIGILES_EXACT_LABEL") == "1" else make_unique_label(_BASE_LABEL)
)
_REPLICATION = int(os.environ.get("VIGILES_REPLICATION") or _EXP.get("replication", 1))
_DURATION_H = float(os.environ.get("VIGILES_DURATION_HOURS") or _EXP.get("duration_hours", 1))


def manifest_sha256(manifest: dict) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> None:
    pkg_digest = compute_package_digest(PKG)  # over executor.py + lite.py

    manifest = {
        "schema_version": "0.1",
        "tenant_id": TENANT_ID,
        "tenant_maintainer_contact": "vigiles@auspexai.network",
        "experiment_id": LABEL,
        "research_goal_paragraph": (
            "Vigiles D6 — the first real tenant: a deterministic behavioral-drift "
            "probe panel run against a worker-served local LLM through the AuspexAI "
            "inference broker. Each work unit re-runs a fixed probe at temperature 0 "
            "with a pinned seed; the executor reduces the response to a sha256 anchor "
            "plus light lexical features (no raw text — Research Ethics §7 "
            "containment) and the adaptive driver folds the per-probe consensus "
            "hashes into a Counter, converging once every probe's consensus response "
            "holds stable across consecutive rounds. Exercises the full local-LLM "
            "execution path (W-S model serving + sandbox inference broker) and the "
            "persisted, Rekor-anchored result-set attestation end-to-end."
        ),
        "models": [{"id": MODEL_ID, "version": "1.0", "local_weights_required": True}],
        "prompt_set_characteristics": (
            "A small fixed panel of neutral instruction-following probes "
            "(greeting, arithmetic, enumeration) carried in each unit's payload; "
            "re-run every round with a constant seed so a changed consensus hash "
            "is the drift signal."
        ),
        "sensitive_content_flags": [],
        "expected_duration_hours": _DURATION_H,
        "replication_factor": _REPLICATION,
        "work_unit_source": {"kind": "static", "tarball_sha256": "0" * 64},
        "executor": {"command": ["python", "executor.py"], "package_sha256": pkg_digest},
        "reducer": {"kind": "builtin_hash_agreement"},
    }

    Manifest.model_validate(manifest)  # fail early on any schema problem
    (PKG / "manifest.json").write_text(json.dumps(manifest, indent=2))
    msha = manifest_sha256(manifest)

    print(f"package_sha256 : {pkg_digest}")
    print(f"manifest_sha256: {msha}")
    print(f"label          : {LABEL}")
    print(f"model_id       : {MODEL_ID}")
    print(f"replication    : {_REPLICATION}")
    print(f"package files  : {sorted(p.name for p in PKG.iterdir())}")
    print()
    print("# --- one step: sign + upload the package + create the experiment ---")
    print("auspexai-tenant experiment submit pkg/ --key <vigiles_key>")
    print("# --- then drive it (the worker AUTO-FETCHES the package; no staging) ---")
    print("cd driver && auspexai-tenant experiment run latest --key <vigiles_key>")
    print(f"# (the GGUF for {MODEL_ID} must already be in the worker's models/ store + served)")


if __name__ == "__main__":
    main()
