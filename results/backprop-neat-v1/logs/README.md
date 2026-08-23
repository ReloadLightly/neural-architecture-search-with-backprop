# v1 shard logs

The console output of the four shards that produced protocol v1. They belong
here rather than at the repository root because they are v1 records, and v1 is
invalidated: **no number in these logs may be cited.**

Provenance was established by matching, not by filename. Each log contains two
segments, in the order `run_shard.sh` executes them — track A, then track B:

| Log | Track-A runs | Track-B runs |
|---|---|---|
| `shard0.log` | 45 | 10 |
| `shard1.log` | 45 | 11 |
| `shard2.log` | 45 | 16 |
| `shard3.log` | 45 | 16 |

Every one of those 180 + 53 = 233 logged runs reproduces the exact
`(validation_accuracy, causal_hidden_nodes, candidate_evaluations)` triple of
the correspondingly named record under `track-a/raw/runs/` and
`track-b/raw/runs/` in this directory's parent — 45/45 and 10/10, 11/11, 16/16,
16/16. That is the whole of v1: 180 track-A and 53 track-B runs, the count
recorded in `INVALIDATED.md`. No segment matches the v2 release.

The logs are retained for the same reason the records are: so that the
invalidation can be audited rather than taken on trust.
