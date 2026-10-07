#!/bin/sh
# Commit and push whatever v6 run records exist, so a reclaimed container costs
# at most one checkpoint interval rather than the whole suite.
#
# R2: push early and often. This session's container is ephemeral and the v6
# ladder takes about five hours, so "often" has to mean "on a timer" rather
# than "when I remember". The suite is resumable by fingerprint, so a pushed
# partial release is a resumable one.
#
# Scoped on purpose: every git command below names
# `results/backprop-neat-v6`, so a checkpoint that fires while other work is
# staged commits the run records and leaves that work alone.
set -eu
cd "$(dirname "$0")/.."
# Every protocol with a live suite, not just v6. The queue runs four phases and
# this script was scoped to the first one's directory, so v8's records would
# have sat uncommitted on an ephemeral container for its whole five-hour suite.
# Caught at 8 runs; it would have been 360.
RUNS="results/backprop-neat-v6 results/backprop-neat-v8"
LOCK=.git/v6-checkpoint.lock

mkdir "$LOCK" 2>/dev/null || { echo "a checkpoint is already running"; exit 0; }
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT INT TERM

N=0
for dir in $RUNS; do
  C=$(ls "$dir"/raw/runs/*.json 2>/dev/null | wc -l | tr -d ' ')
  N=$((N + C))
done
[ "$N" -gt 0 ] || { echo "no records yet"; exit 0; }

# shellcheck disable=SC2086
git add -- $RUNS >/dev/null
# shellcheck disable=SC2086
if git diff --cached --quiet -- $RUNS; then
  echo "no change at $N records"
  exit 0
fi

PLANNED=$(.venv/bin/python -c \
  "from bpneat.v6.protocol import planned_runs as a
from bpneat.v8.protocol import planned_runs as b
print(len(a()) + len(b()))")
git commit -q -m "run checkpoint: $N/$PLANNED confirmatory runs across the live protocols

Partial raw records, pushed on a timer. The suite is resumable by fingerprint,
so this is a checkpoint and not a release: no derived table, no sealed test, no
claim. See docs/v6-preregistration.md for what is being run.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GP1bnJXe8xy12aBhsum7ab" -- $RUNS

for i in 1 2 3 4; do
  git push -q origin main && break || sleep $((2 ** i))
done
git branch -f claude/backprop-neat-actir-submission-kz5w2v main
for i in 1 2 3 4; do
  git push -q origin claude/backprop-neat-actir-submission-kz5w2v && break || sleep $((2 ** i))
done
echo "checkpointed $N/$PLANNED at $(date -u +%H:%M:%SZ)"
