# Vigiles — AuspexAI's first tenant

Vigiles is the first research tenant on the [AuspexAI](https://github.com/auspexai) volunteer compute network. It is an AuspexAI-native clean rebuild for multi-agent LLM behavioral drift research, designed against the [Tenant SDK](https://github.com/auspexai/tenant-sdk) contract from day one.

Vigiles is a sibling project to [Sentinel](https://github.com/jasongagne-git/sentinel), sharing methodological DNA (behavioral drift research) but maintaining its own codebase and roadmap. Sentinel continues to evolve as an independent research program; Vigiles is not a fork or adaptation of Sentinel code.

## Status

**Not yet started.** Vigiles is sequenced last in Phase 1 (per [Principles & Scope §5.3](https://github.com/auspexai/platform)) so the platform interfaces prove themselves against the synthetic test tenant before bending toward a real tenant's needs. The first experiment will be **D6** — a week-long continuous multi-agent drift run.

## Scope

This repository will hold:

- The Vigiles tenant package — executor and reducer consuming the AuspexAI [Tenant SDK](https://github.com/auspexai/tenant-sdk)
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
