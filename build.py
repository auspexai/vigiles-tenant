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

from auspexai_tenant.manifest import Manifest, compute_package_digest

HERE = Path(__file__).parent
PKG = HERE / "pkg"  # executor.py + lite.py (both digested into package_sha256)
TENANT_ID = os.environ.get(
    "VIGILES_TENANT_ID", "vigiles-lab"
)  # the onboarded tenancy (legacy hand-created "vigiles" retired 2026-06-12)
# Per-tenant experiment labels are unique FOREVER (aborted runs keep theirs) —
# override for re-runs: VIGILES_LABEL=vigiles-d6-drift-r2 python build.py
LABEL = os.environ.get("VIGILES_LABEL", "vigiles-d6-drift")
# Default to a small instruct GGUF that fits an 8GB Jetson; override to match
# the worker's served store id exactly (the #30 routing key).
MODEL_ID = os.environ.get("VIGILES_MODEL_ID", "gemma-3-1b-it-q4")


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
        "expected_duration_hours": 1,
        "replication_factor": 1,
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
    print(f"package files  : {sorted(p.name for p in PKG.iterdir())}")
    print()
    print("# --- sign (writes pkg/manifest.json.sig; now excluded from the digest) ---")
    print(
        "auspexai-tenant manifest sign pkg/manifest.json --key <vigiles_key> -o pkg/manifest.json.sig"
    )
    print("# --- stage on the serving worker (operator) ---")
    print(f"ssh <worker> 'mkdir -p ~/.local/share/auspexai-worker/tenants/{msha}'")
    print(
        f"scp {PKG}/manifest.json {PKG}/executor.py {PKG}/lite.py <worker>:~/.local/share/auspexai-worker/tenants/{msha}/"
    )
    print(f"# (the GGUF for {MODEL_ID} must already be in the worker's models/ store + served)")


if __name__ == "__main__":
    main()
