# Published Drift-Benchmark entries

Signed registry entries bound for the public board (auspexai.network/benchmarks.html).

Flow (drift_benchmark_design.md §6/§8 — publishing is a deliberate act):

1. Score exists (automatic at launch, or `auspexai-tenant benchmark drift`).
2. `auspexai-tenant benchmark publish <label-or-exp-id>` — re-exports and
   custody-verifies BOTH bundles, then signs the claim with this tenancy's key.
3. Commit the emitted `benchmark_entry_<reference>.json` HERE and push.
4. The board curator verifies (`benchmark verify-entry`), grounds the publisher
   key to this tenancy, adds the entry to the site registry, and deploys.

Inclusion is curated, never automatic. Every entry is independently verifiable
by anyone — signature, attestation Merkle roots, and Rekor inclusion ride in
the entry itself.
