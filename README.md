# Sentinel Tenant for AuspexAI

[Sentinel](https://github.com/jasongagne-git/sentinel) — a multi-agent LLM behavioral drift research program — packaged as the first tenant on the [AuspexAI](https://github.com/auspexai) network.

## Status

**Phase 0 — Foundation.** Tenant packaging begins in Phase 1, alongside the [platform](https://github.com/auspexai/platform) code that makes hosting possible. The first experiment to run via this packaging is **D6** — a week-long continuous multi-agent run with rolling worker membership.

## Scope

This repository will hold:

- The Sentinel tenant package — project module consuming the AuspexAI [Tenant SDK](https://github.com/auspexai/tenant-sdk)
- Experiment manifests and result schemas specific to Sentinel research
- Containment plan implementations per the [AuspexAI Research Ethics Policy](https://github.com/auspexai/.github/blob/main/RESEARCH_ETHICS_POLICY.md) §7

The upstream research code lives in [`jasongagne-git/sentinel*`](https://github.com/jasongagne-git?tab=repositories&q=sentinel) under Apache-2.0; this repository contains the AuspexAI-specific packaging and integration layer, not duplication of the science code.

## Research ethics

Sentinel is the worked example in the AuspexAI Research Ethics Policy (§7). Classification: **medium dual-use risk**. The research subject is behavioral drift — including drift toward harmful behavior such as toxicity, manipulation, or destabilized agent identity. Harmful outputs are research evidence with documented containment, not deliverables. Public reporting uses redacted excerpts, aggregated metrics, and synthetic illustrations of drift patterns; raw transcripts are not publicly released by default.

Until the Approver pool seats with floor of 2 external members, Sentinel approval proceeds under the Maintainer-as-tenant-author procedure — see [`GOVERNANCE.md`](https://github.com/auspexai/.github/blob/main/GOVERNANCE.md) §6.3.

## License

[Apache-2.0](LICENSE) — matching the upstream Sentinel research code license. Per the AuspexAI Tenant SDK boundary design, tenant license is the Researcher's choice; this tenant uses Apache-2.0 to remain aligned with the upstream science and to demonstrate that the AGPL-3.0 platform genuinely supports non-AGPL tenants.

## Contributing

See [`CONTRIBUTING.md`](https://github.com/auspexai/.github/blob/main/CONTRIBUTING.md) (org-wide). For the Sentinel research science itself — datasets, experimental design, statistical methods — see the upstream repositories. This repository is for the AuspexAI-tenant packaging and integration only.

## Governance

Project direction is held by the Maintainer team per [`GOVERNANCE.md`](https://github.com/auspexai/.github/blob/main/GOVERNANCE.md). Tenant approval and revocation authority rests with the Approver pool. Code of Conduct: [`CODE_OF_CONDUCT.md`](https://github.com/auspexai/.github/blob/main/CODE_OF_CONDUCT.md).

## Watch this repo

Activity will begin as Phase 1 ramps up.
