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
COMMITTED=no
if git diff --cached --quiet -- $RUNS; then
  echo "no change at $N records"
else
  COMMITTED=yes
fi

PLANNED=$(.venv/bin/python -c \
  "from bpneat.v6.protocol import planned_runs as a
from bpneat.v8.protocol import planned_runs as b
print(len(a()) + len(b()))")
[ "$COMMITTED" = no ] || git commit -q -m "run checkpoint: $N/$PLANNED confirmatory runs across the live protocols

Partial raw records, pushed on a timer. The suite is resumable by fingerprint,
so this is a checkpoint and not a release: no derived table, no sealed test, no
claim. See docs/v6-preregistration.md for what is being run.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GP1bnJXe8xy12aBhsum7ab" -- $RUNS

# Push whenever the branch is ahead of the remote -- not only when this run
# had something new to commit.
#
# The "no change" case used to `exit 0` right here, which made an unpushed
# commit permanent: a push that failed (a dropped network, a harness restart
# between the commit and the push) left the commit local, and every later run
# found nothing new to commit and exited before reaching the push. Six commits
# had piled up behind that before a git hook noticed. The records were safe on
# disk the whole time, but "pushed early and often" had quietly become "pushed
# until the first failure".
#
# A failed push is also no longer silent. The old loop was
# `git push && break || sleep`, whose `||` keeps `set -e` from firing, so four
# failed attempts printed nothing and returned success.
push_branch() {   # push_branch <branch>
  for i in 1 2 3 4; do
    git push -q origin "$1" && return 0
    sleep $((2 ** i))
  done
  echo "PUSH FAILED for $1 after 4 attempts; $(git rev-list --count "origin/$1..$1") commit(s) still local" >&2
  return 1
}

FAILED=no
if [ "$(git rev-list --count origin/main..main)" -gt 0 ]; then
  push_branch main || FAILED=yes
fi
git branch -f claude/backprop-neat-actir-submission-kz5w2v main
if [ "$(git rev-list --count \
        origin/claude/backprop-neat-actir-submission-kz5w2v..claude/backprop-neat-actir-submission-kz5w2v)" -gt 0 ]; then
  push_branch claude/backprop-neat-actir-submission-kz5w2v || FAILED=yes
fi

if [ "$FAILED" = yes ]; then
  echo "checkpoint at $N records: committed but NOT fully pushed at $(date -u +%H:%M:%SZ)"
  exit 1
fi
if [ "$COMMITTED" = yes ]; then
  echo "checkpointed $N/$PLANNED at $(date -u +%H:%M:%SZ)"
else
  echo "nothing new at $N records; remote up to date at $(date -u +%H:%M:%SZ)"
fi
