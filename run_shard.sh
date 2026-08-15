#!/bin/sh
# One shard: track A then track B, same shard index. Resumable and idempotent.
i="$1"; n="$2"
export PYTHONPATH=src
for track in A B; do
  case "$track" in A) d=track-a;; B) d=track-b;; esac
  .venv/bin/python -u -m bpneat.cli suite \
    --track "$track" \
    --out "results/backprop-neat-v2/$d" \
    --shard-index "$i" --shard-total "$n"
done
