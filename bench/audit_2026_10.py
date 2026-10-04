"""Reproduce the October 2026 external audit from the committed v2 release.

Findings F1-F4 are recomputed from ``raw/runs/*.json`` and ``final-test.json``;
F1's learner probe is re-run live on the **validation** split only, so this
script never touches a sealed test partition.

Run: ``python bench/audit_2026_10.py``  (add ``--fast`` to skip the 20k-step arm)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bpneat.baselines import make_mlp  # noqa: E402
from bpneat.datasets import make_bundle  # noqa: E402
from bpneat.genome import OP_SIN, OP_TANH, backward, forward  # noqa: E402
from bpneat.learn import _sigmoid, accuracy, train  # noqa: E402

V2 = ROOT / "results" / "backprop-neat-v2"
OUT: list[str] = []


def say(line: str = "") -> None:
    print(line, flush=True)
    OUT.append(line)


def load(track: str) -> tuple[list[dict], dict[str, dict]]:
    runs = [json.loads(p.read_text()) for p in sorted((V2 / track / "raw" / "runs").glob("*.json"))]
    final = {r["run_id"]: r for r in json.loads((V2 / track / "final-test.json").read_text())["results"]}
    return runs, final


# ---------------------------------------------------------------- F1

def f1_budgets(runs: list[dict]) -> bool:
    say("## F1  Starved controls")
    say()
    say("Realized gradient steps per run, mean over 10 replicates (track B):")
    say()
    say("| task | condition | candidates | gradient steps | steps/restart |")
    say("|---|---|---|---:|---:|")
    by = defaultdict(list)
    for r in runs:
        by[(r["task"], r["condition"])].append(r)
    hold = True
    for task in ("xor", "circle", "spiral"):
        for cond in ("backprop_neat", "fixed_mlp", "random_search"):
            items = by[(task, cond)]
            if not items:
                continue
            cands = statistics.mean(i["compute"]["candidate_evaluations"] for i in items)
            steps = statistics.mean(i["compute"]["gradient_steps"] for i in items)
            per = steps / cands if cands else float("nan")
            say(f"| {task} | {cond} | {cands:.0f} | {steps:,.0f} | {per:.1f} |")
    say()
    sp_mlp = statistics.mean(i["compute"]["gradient_steps"] for i in by[("spiral", "fixed_mlp")])
    sp_bpn = statistics.mean(i["compute"]["gradient_steps"] for i in by[("spiral", "backprop_neat")])
    sp_per = sp_mlp / 60
    say(f"Spirals: fixed_mlp {sp_mlp:,.0f} total steps over 60 restarts = {sp_per:.0f} per restart;")
    say(f"backprop_neat {sp_bpn:,.0f}. Ratio {sp_bpn / sp_mlp:.0f}x.")
    ok = abs(sp_mlp - 2506) < 400 and abs(sp_bpn - 132138) < 20000 and sp_per < 60
    say(f"Audit claimed ~2,506 / ~42 per restart / 132,138. **{'REPRODUCES' if ok else 'DOES NOT REPRODUCE'}**")
    hold &= ok
    say()
    rs = statistics.mean(i["compute"]["candidate_evaluations"] for i in by[("spiral", "random_search")])
    ev = statistics.mean(i["compute"]["candidate_evaluations"] for i in by[("spiral", "backprop_neat")])
    ok2 = rs == 60 and ev >= 2000
    say(f"random_search gets {rs:.0f} candidates vs {ev:.0f} for evolution. **{'REPRODUCES' if ok2 else 'DOES NOT REPRODUCE'}**")
    say()
    return hold and ok2


def f1_probe(fast: bool) -> bool:
    say("### F1 probe — the learner, not the architecture")
    say()
    say("Validation split only; the sealed test is never read here.")
    say()
    seeds = [8101, 8202, 8303, 8404]

    def plain_train(g, X, y, n_steps, seed, batch=10, lr=0.01, decay=0.99):
        """Plain RMSProp, no rollback, no early break."""
        rng = np.random.default_rng(seed)
        w = np.array(g.weight, dtype=np.float64)
        cache = np.zeros(len(w))
        for _ in range(n_steps):
            idx = rng.integers(0, len(X), batch)
            tape = forward(g, X[idx], w, settle=True)
            d = (_sigmoid(tape.vals[tape.out_var]) - y[idx]) / batch
            grad = np.clip(backward(tape, d, w), -5.0, 5.0)
            cache *= decay
            cache += (1 - decay) * grad * grad
            w -= lr * grad / np.sqrt(np.maximum(cache, 1e-8)) + 0.001 * w
            np.clip(w, -50, 50, out=w)
        return w

    rows = []
    for seed in seeds:
        b = make_bundle("spiral", seed=seed)
        Xt, yt, Xv, yv = b.train.X, b.train.y, b.validation.X, b.validation.y

        g = make_mlp((32, 32), np.random.default_rng(1), op=OP_TANH)
        res = train(g, np.array(g.weight), Xt, yt, np.random.default_rng(2), n_cycles=600, settle=True)
        ha_steps, ha_acc = res.gradient_steps, accuracy(g, res.weights, Xv, yv, True)

        g2 = make_mlp((32, 32), np.random.default_rng(1), op=OP_TANH)
        n_long = 2000 if fast else 20000
        w2 = plain_train(g2, Xt, yt, n_long, seed=2)
        long_acc = accuracy(g2, w2, Xv, yv, True)

        g3 = make_mlp((32, 32), np.random.default_rng(1), op=OP_SIN)
        w3 = plain_train(g3, Xt, yt, 600, seed=2)
        sin_acc = accuracy(g3, w3, Xv, yv, True)

        rows.append((seed, ha_steps, ha_acc, long_acc, sin_acc))
        say(f"  seed {seed}: Ha learner {ha_steps:3d} steps -> {ha_acc:.3f} | "
            f"plain {n_long} steps -> {long_acc:.3f} | sin-MLP 600 steps -> {sin_acc:.3f}")
    say()
    ha = [r[2] for r in rows]
    lg = [r[3] for r in rows]
    sn = [r[4] for r in rows]
    st = [r[1] for r in rows]
    say(f"| arm | steps | val acc range |")
    say(f"|---|---:|---|")
    say(f"| 32x32 tanh, Ha learner | {min(st)}-{max(st)} | {min(ha):.2f}-{max(ha):.2f} |")
    say(f"| 32x32 tanh, plain {n_long} steps | {n_long} | {min(lg):.2f}-{max(lg):.2f} |")
    say(f"| 32x32 sin, plain 600 steps | 600 | {min(sn):.2f}-{max(sn):.2f} |")
    say()
    ok = max(ha) < 0.70 and min(lg) > max(ha) and min(sn) > max(ha)
    say(f"The same architecture crosses from near-chance to competent by changing only the")
    say(f"stopping rule. **{'REPRODUCES' if ok else 'DOES NOT REPRODUCE'}**")
    say()
    return ok


# ---------------------------------------------------------------- F2

def f2_xor(runs: list[dict], final: dict[str, dict]) -> bool:
    say("## F2  The XOR text contradicts the table")
    say()
    bpn = {r["replicate"]: final[r["run_id"]] for r in runs
           if r["task"] == "xor" and r["condition"] == "backprop_neat"}
    mlp = {r["replicate"]: final[r["run_id"]] for r in runs
           if r["task"] == "xor" and r["condition"] == "fixed_mlp"}
    reps = sorted(bpn)
    acc_win = sum(bpn[r]["test_accuracy"] > mlp[r]["test_accuracy"] for r in reps)
    acc_tie = sum(bpn[r]["test_accuracy"] == mlp[r]["test_accuracy"] for r in reps)
    loss_win = sum(bpn[r]["test_loss"] < mlp[r]["test_loss"] for r in reps)
    perfect = sum(bpn[r]["test_accuracy"] == 1.0 for r in reps)
    losses = sorted(bpn[r]["test_loss"] for r in reps)
    say("| replicate | BPN acc | MLP acc | BPN loss | MLP loss |")
    say("|---|---:|---:|---:|---:|")
    for r in reps:
        say(f"| {r} | {bpn[r]['test_accuracy']:.3f} | {mlp[r]['test_accuracy']:.3f} | "
            f"{bpn[r]['test_loss']:.3f} | {mlp[r]['test_loss']:.3f} |")
    say()
    say(f"- Backprop-NEAT reaches test accuracy 1.000 in **{perfect}/10** replicates.")
    say(f"- It beats fixed_mlp on accuracy in **{acc_win}/10** (ties {acc_tie}), on loss in **{loss_win}/10**.")
    say(f"- Mean BCE {statistics.mean(losses):.3f} vs median {statistics.median(losses):.3f}; "
        f"two outliers at {losses[-2]:.2f} and {losses[-1]:.2f}.")
    ok = perfect == 7 and (acc_win + acc_tie) == 10 and loss_win == 8
    say(f"Audit claimed 7/10 perfect, 10/10 on accuracy, 8/10 on loss. "
        f"**{'REPRODUCES' if ok else 'DOES NOT REPRODUCE'}**")
    say()
    return ok


# ---------------------------------------------------------------- F3

def f3_confound() -> bool:
    say("## F3  The propagation claim is confounded")
    say()
    a = json.loads((V2 / "track-a" / "raw" / "runs" / "xor__backprop_neat__r01.json").read_text())
    b = json.loads((V2 / "track-b" / "raw" / "runs" / "xor__backprop_neat__r01.json").read_text())
    say("| | track A | track B |")
    say("|---|---|---|")
    say(f"| propagation | `{a['config']['propagation']}` | `{b['config']['propagation']}` |")
    say(f"| fitness split | `{a['config']['fitness_split']}` | `{b['config']['fitness_split']}` |")
    say()
    ok = (a["config"]["propagation"] != b["config"]["propagation"]
          and a["config"]["fitness_split"] != b["config"]["fitness_split"])
    say("The tracks differ in **two** factors at once, so no difference between them can be")
    say(f"attributed to propagation alone. **{'REPRODUCES' if ok else 'DOES NOT REPRODUCE'}**")
    say()
    return ok


# ---------------------------------------------------------------- F4

def f4_collapse(runs_a: list[dict], final_a: dict[str, dict]) -> bool:
    say("## F4  The retraction went too far")
    say()
    hold = True
    for task in ("xor", "spiral"):
        items = [r for r in runs_a if r["task"] == task and r["condition"] == "backprop_neat"]
        zero = [r for r in items if r["metrics"]["causal_hidden_nodes"] == 0]
        say(f"**{task}** (track A, backprop_neat): "
            f"{len(zero)}/{len(items)} champions have 0 causally active hidden nodes.")
        if zero:
            ids = ", ".join(f"r{r['replicate']:02d}" for r in sorted(zero, key=lambda x: x["replicate"]))
            accs = [final_a[r["run_id"]]["test_accuracy"] for r in zero]
            say(f"  collapsed replicates: {ids} — test accuracy "
                f"{min(accs):.3f}-{max(accs):.3f} (logistic floor)")
        live = [final_a[r["run_id"]]["test_accuracy"] for r in items
                if r["metrics"]["causal_hidden_nodes"] > 0]
        if live:
            say(f"  non-collapsed: test accuracy {min(live):.3f}-{max(live):.3f}")
        mean = statistics.mean(final_a[r["run_id"]]["test_accuracy"] for r in items)
        say(f"  mean test accuracy {mean:.3f} — a **bimodal** mixture, not a central tendency")
        say()
        if task == "xor":
            hold &= len(zero) == 3
        else:
            hold &= len(zero) == 1
    say(f"Audit claimed 3/10 XOR and 1/10 spirals collapse. "
        f"**{'REPRODUCES' if hold else 'DOES NOT REPRODUCE'}**")
    say()
    return hold


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="2k instead of 20k steps in the probe")
    args = ap.parse_args()

    started = time.time()
    runs_a, final_a = load("track-a")
    runs_b, final_b = load("track-b")

    say("# Audit reproduction — October 2026")
    say()
    say("Recomputed from the committed v2 release by `bench/audit_2026_10.py`.")
    say("Every number below is derived from `raw/runs/*.json` and `final-test.json`;")
    say("the F1 probe trains live on the **validation** split only.")
    say()
    say(f"- {len(runs_a)} track-A records, {len(runs_b)} track-B records")
    say(f"- fingerprint `cfdf1fa3198adc0e369466800ede0ea5afa24d99af18000a2161357ae5ec0d75`")
    say()

    results = {
        "F1 budgets": f1_budgets(runs_b),
        "F1 probe": f1_probe(args.fast),
        "F2 xor": f2_xor(runs_b, final_b),
        "F3 confound": f3_confound(),
        "F4 collapse": f4_collapse(runs_a, final_a),
    }

    say("## Verdict")
    say()
    say("| finding | reproduces |")
    say("|---|---|")
    for k, v in results.items():
        say(f"| {k} | {'yes' if v else '**NO**'} |")
    say()
    say(f"Elapsed {time.time() - started:.0f}s.")

    target = ROOT / "docs" / "audit-2026-10.md"
    target.parent.mkdir(exist_ok=True)
    target.write_text("\n".join(OUT) + "\n")
    print(f"\nwrote {target}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
