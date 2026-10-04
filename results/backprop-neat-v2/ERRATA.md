# Errata for this release

These runs are correct and regenerate byte-identically from their raw records.
What was wrong is what was **claimed** about them.

An October 2026 audit showed the headline comparison is confounded: the
fixed-MLP control received 2,506 realized gradient steps on spirals against
Backprop-NEAT's 132,138, because Ha's rollback rule halts a 69-node network
after ~42 updates. Two v2 claims are withdrawn, two narrowed.

Full text: [`../../docs/v2-errata.md`](../../docs/v2-errata.md).
Reproduction: [`../../docs/audit-2026-10.md`](../../docs/audit-2026-10.md).

Do not cite this release's between-condition comparisons against *fixed* or
*randomly sampled* architectures. The two evolutionary ablations
(`homogeneous_tanh`, `evolution_only`) share the candidate budget by
construction and remain usable.
