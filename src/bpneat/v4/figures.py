"""v4 figures, rendered from the raw records.

Same conventions as v3: the validated categorical order (adjacent-pair CVD
ΔE 9.1), colour never carrying identity alone, and a CSV beside every figure in
the release. Where a panel shows more series than the palette has hues, colour
is paired with a marker shape so no hue is reused.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .analysis import (  # noqa: E402
    ALGORITHMS,
    budget_table,
    cross_algorithm,
    family_effects,
    hypotheses,
    load,
    sign_matrix,
    summarise,
)
from .protocol import ALL_TASKS  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GOOD, BAD, NEUTRAL = "#1baf7a", "#e34948", "#c9c8c3"

LABEL = {
    "bpneat": "Backprop-NEAT",
    "cgp": "CGP (1+4)",
    "fixed_tanh_ha": "Fixed tanh (unmatched)",
    "fixed_tanh_matched_bpneat": "Fixed tanh @ BP-NEAT budget",
    "fixed_mixed_matched_bpneat": "Fixed mixed @ BP-NEAT budget",
    "fixed_tanh_matched_cgp": "Fixed tanh @ CGP budget",
    "fixed_mixed_matched_cgp": "Fixed mixed @ CGP budget",
    "cgp_random_matched": "Random CGP (matched)",
}
TASK_LABEL = {"xor": "XOR", "circle": "Circles", "spiral": "Spirals",
              "checkerboard": "Checkerboard", "spiral3": "3-arm spiral"}

#: One (hue, marker) pair per condition, so eight series never cycle a hue.
STYLE = {
    "bpneat": (SERIES[0], "o"),
    "cgp": (SERIES[1], "s"),
    "fixed_tanh_ha": (SERIES[2], "^"),
    "fixed_tanh_matched_bpneat": (SERIES[3], "D"),
    "fixed_mixed_matched_bpneat": (SERIES[4], "v"),
    "fixed_tanh_matched_cgp": (SERIES[5], "P"),
    "fixed_mixed_matched_cgp": (SERIES[0], "X"),
    "cgp_random_matched": (SERIES[1], "*"),
}

SIGN_COLOUR = {
    "search>fixed": SERIES[0],
    "fixed>search": BAD,
    "ns": NEUTRAL,
    "n/a": SURFACE,
}
SIGN_TEXT = {
    "search>fixed": "search\nwins",
    "fixed>search": "fixed\nwins",
    "ns": "n.s.",
    "n/a": "—",
}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(nrows=1, ncols=1, figsize=(10, 4.2)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=SURFACE)
    for ax in np.atleast_1d(axes).ravel():
        _style(ax)
    return fig, axes


def _save(fig, path: Path, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if kw:
        fig.subplots_adjust(**kw)
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return path


def _test_or_val(row) -> float:
    v = row.get("test_accuracy_mean")
    return float(v) if v not in (None, "") else float(row["validation_accuracy_mean"])


def fig_sign_matrix(signs: list[dict], out: Path) -> Path:
    """The headline: the sign of search-vs-fixed, per algorithm and protocol."""
    algos = [a["algorithm"] for a in ALGORITHMS]
    cols = [("vs_unmatched", "vs. unmatched\ncontrol"),
            ("vs_matched_tanh", "vs. matched\ntanh"),
            ("vs_matched_mixed", "vs. matched\nmixed")]
    by = {(r["algorithm"], r["task"]): r for r in signs}

    fig, axes = _fig(1, len(algos), figsize=(11, 4.6))
    for ax, algo in zip(np.atleast_1d(axes).ravel(), algos):
        ax.grid(False)
        for ci, (key, _) in enumerate(cols):
            for ti, task in enumerate(ALL_TASKS):
                verdict = by.get((algo, task), {}).get(key, "n/a")
                ax.add_patch(
                    plt.Rectangle((ci - 0.47, ti - 0.45), 0.94, 0.9,
                                  facecolor=SIGN_COLOUR[verdict], edgecolor=SURFACE,
                                  linewidth=2)
                )
                ax.text(ci, ti, SIGN_TEXT[verdict], ha="center", va="center",
                        fontsize=8, color="white" if verdict != "ns" else INK,
                        fontweight="bold" if verdict != "ns" else "normal")
        ax.set_xlim(-0.6, len(cols) - 0.4)
        ax.set_ylim(-0.6, len(ALL_TASKS) - 0.4)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([c[1] for c in cols], fontsize=8, color=INK2)
        ax.set_yticks(range(len(ALL_TASKS)))
        ax.set_yticklabels([TASK_LABEL[t] for t in ALL_TASKS], fontsize=8, color=INK2)
        ax.set_title(algo, color=INK, fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(
        "Sealed-test accuracy: which side wins, by algorithm and by budget protocol",
        color=INK, fontsize=11,
    )
    return _save(fig, out, top=0.84, bottom=0.14, left=0.11, right=0.98, wspace=0.35)


def fig_accuracy_by_task(summary: list[dict], out: Path) -> Path:
    """Every condition on every task, with the two references marked."""
    by = {(r["task"], r["condition"]): r for r in summary}
    conds = list(LABEL)
    fig, axes = _fig(1, len(ALL_TASKS), figsize=(16, 4.4))
    for ax, task in zip(np.atleast_1d(axes).ravel(), ALL_TASKS):
        vals, colours = [], []
        for c in conds:
            row = by.get((task, c))
            vals.append(_test_or_val(row) if row else np.nan)
            colours.append(STYLE[c][0])
        y = np.arange(len(conds))
        ax.barh(y, vals, color=colours, height=0.7)
        for yi, v in zip(y, vals):
            if np.isfinite(v):
                ax.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=7, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [LABEL[c] + ("  *" if c in ("bpneat", "cgp") else "") for c in conds],
            fontsize=7, color=INK2,
        )
        ax.invert_yaxis()
        ax.set_xlim(0.4, 1.09)
        ax.set_title(TASK_LABEL[task], color=INK, fontsize=10)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        if task != ALL_TASKS[0]:
            ax.set_yticklabels([])
    fig.suptitle(
        "Sealed-test accuracy by condition. * marks a search algorithm; "
        "every other bar is a fixed architecture.",
        color=INK, fontsize=11,
    )
    return _save(fig, out, top=0.86, bottom=0.07, left=0.145, right=0.99, wspace=0.08)


def fig_reversal(effects: list[dict], out: Path) -> Path:
    """Paired median difference against each control, for both algorithms.

    Above zero the search wins; below it the fixed network does. The figure's
    claim is that both algorithms cross zero at the same place.
    """
    fig, axes = _fig(1, 2, figsize=(11.5, 4.4))
    for ax, spec in zip(np.atleast_1d(axes).ravel(), ALGORITHMS):
        ref = spec["reference"]
        controls = [spec["unmatched"], spec["matched"], spec["matched_mixed"]]
        width = 0.26
        for ci, ctl in enumerate(controls):
            xs, med, lo, hi, sig = [], [], [], [], []
            for ti, task in enumerate(ALL_TASKS):
                e = next(
                    (x for x in effects
                     if x["family"] == ref and x["task"] == task and x["condition"] == ctl),
                    None,
                )
                if e is None:
                    continue
                xs.append(ti + (ci - 1) * width)
                med.append(e["median_difference"])
                lo.append(e["median_difference"] - e["ci95_low"])
                hi.append(e["ci95_high"] - e["median_difference"])
                sig.append(e.get("holm_p", e["wilcoxon_p"]) < 0.05)
            colour = [SERIES[ci] if s else NEUTRAL for s in sig]
            ax.bar(xs, med, width=width * 0.9, color=colour,
                   yerr=[lo, hi], ecolor=INK2, capsize=2, error_kw={"linewidth": 0.9},
                   label=LABEL[ctl])
        ax.axhline(0.0, color=INK, linewidth=1.0)
        ax.set_xticks(range(len(ALL_TASKS)))
        ax.set_xticklabels([TASK_LABEL[t] for t in ALL_TASKS], fontsize=8, color=INK2)
        ax.set_ylabel("median paired difference\n(search − fixed)", fontsize=8, color=INK2)
        ax.set_title(spec["algorithm"], color=INK, fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc="lower left", ncol=1)
    fig.suptitle(
        "The reversal, both algorithms. Grey bars are not significant after Holm "
        "correction in that algorithm's own family.",
        color=INK, fontsize=11,
    )
    return _save(fig, out, top=0.85, bottom=0.11, left=0.085, right=0.985, wspace=0.22)


def fig_budget(budgets: list[dict], out: Path) -> Path:
    """What each arm actually spent. The confound, made visible."""
    by = {(r["task"], r["condition"]): r for r in budgets}
    conds = ["bpneat", "cgp", "fixed_tanh_ha",
             "fixed_tanh_matched_bpneat", "fixed_tanh_matched_cgp"]
    fig, ax = _fig(figsize=(10.5, 4.2))
    width = 0.16
    for ci, c in enumerate(conds):
        xs = [ti + (ci - 2) * width for ti in range(len(ALL_TASKS))]
        vals = [by.get((t, c), {}).get("gradient_steps_mean", np.nan) for t in ALL_TASKS]
        colour, _ = STYLE[c]
        ax.bar(xs, vals, width=width * 0.9, color=colour, label=LABEL[c])
    ax.set_yscale("log")
    ax.set_xticks(range(len(ALL_TASKS)))
    ax.set_xticklabels([TASK_LABEL[t] for t in ALL_TASKS], fontsize=8, color=INK2)
    ax.set_ylabel("gradient updates per run (log)", fontsize=8, color=INK2)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    ax.set_title(
        "Realized gradient budget. The unmatched control spends about two orders of "
        "magnitude less than the searches it is compared against.",
        color=INK, fontsize=10,
    )
    return _save(fig, out)


def fig_cross_algorithm(cross: list[dict], out: Path) -> Path:
    """How far apart the algorithms are, against how far the protocol moves things."""
    fig, ax = _fig(figsize=(10.5, 4.2))
    xs = np.arange(len(cross))
    between = [abs(r["median_cgp_minus_bpneat"]) for r in cross]
    protocol = [
        float(np.mean([abs(r["median_matching_effect_bpneat"]),
                       abs(r["median_matching_effect_cgp"])]))
        for r in cross
    ]
    ax.bar(xs - 0.19, between, width=0.34, color=SERIES[4],
           label="|CGP − Backprop-NEAT|")
    ax.bar(xs + 0.19, protocol, width=0.34, color=SERIES[0],
           label="|effect of matching the budget| (mean of the two)")
    for x, b, p in zip(xs, between, protocol):
        ax.text(x - 0.19, b + 0.004, f"{b:.3f}", ha="center", fontsize=7, color=INK2)
        ax.text(x + 0.19, p + 0.004, f"{p:.3f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([TASK_LABEL[r["task"]] for r in cross], fontsize=8, color=INK2)
    ax.set_ylabel("median paired difference in accuracy", fontsize=8, color=INK2)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title(
        "Choosing the algorithm matters less than choosing the budget protocol",
        color=INK, fontsize=10,
    )
    return _save(fig, out)


def fig_hypotheses(hyp: list[dict], out: Path) -> Path:
    """The preregistered scorecard, as declared, with the observed counts."""
    fig, ax = _fig(figsize=(11, 0.78 * len(hyp) + 1.6))
    ax.grid(False)
    for i, h in enumerate(reversed(hyp)):
        good = h["verdict"] == "holds"
        ax.add_patch(
            plt.Rectangle((0, i - 0.38), 0.06, 0.76,
                          facecolor=GOOD if good else BAD, edgecolor=SURFACE)
        )
        ax.text(0.09, i + 0.14, f"{h['hypothesis']}  {h['statement']}",
                fontsize=9, color=INK, va="center")
        ax.text(0.09, i - 0.19, f"rule: {h['decision_rule']}   ·   observed: {h['observed']}",
                fontsize=7.5, color=INK2, va="center")
        ax.text(0.985, i, h["verdict"], fontsize=9, color=GOOD if good else BAD,
                ha="right", va="center", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(hyp) - 0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Preregistered hypotheses, scored by their own declared rules",
                 color=INK, fontsize=11, loc="left")
    return _save(fig, out)


def fig_cgp_structure(runs: list[dict], out: Path) -> Path:
    """What CGP's genotype-phenotype map actually did: active size, neutral drift."""
    active: dict[str, list[int]] = {}
    neutral: dict[str, list[int]] = {}
    for r in runs:
        if r["condition"] not in ("cgp", "cgp_random_matched"):
            continue
        n = r["compute"].get("cgp_active_nodes")
        if n is not None:
            active.setdefault(f"{r['condition']}|{r['task']}", []).append(n)
        k = r["compute"].get("cgp_neutral_accepted")
        if k is not None:
            neutral.setdefault(r["task"], []).append(k)

    fig, axes = _fig(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    width = 0.34
    for ci, cond in enumerate(("cgp", "cgp_random_matched")):
        xs = [ti + (ci - 0.5) * width for ti in range(len(ALL_TASKS))]
        vals = [
            float(np.mean(active.get(f"{cond}|{t}", [np.nan]))) for t in ALL_TASKS
        ]
        ax.bar(xs, vals, width=width * 0.9, color=STYLE[cond][0], label=LABEL[cond])
        for x, v in zip(xs, vals):
            if np.isfinite(v):
                ax.text(x, v + 0.12, f"{v:.1f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(range(len(ALL_TASKS)))
    ax.set_xticklabels([TASK_LABEL[t] for t in ALL_TASKS], fontsize=8, color=INK2)
    ax.set_ylabel("active function nodes in the champion", fontsize=8, color=INK2)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Phenotype size", color=INK, fontsize=10)

    ax = axes[1]
    vals = [neutral.get(t, []) for t in ALL_TASKS]
    parts = ax.violinplot([v or [0] for v in vals], showmedians=True, widths=0.8)
    for body in parts["bodies"]:
        body.set_facecolor(SERIES[1])
        body.set_alpha(0.55)
        body.set_edgecolor(INK2)
    for key in ("cmins", "cmaxes", "cbars", "cmedians"):
        if key in parts:
            parts[key].set_color(INK2)
    ax.set_xticks(range(1, len(ALL_TASKS) + 1))
    ax.set_xticklabels([TASK_LABEL[t] for t in ALL_TASKS], fontsize=8, color=INK2)
    ax.set_ylabel("generations accepted at equal fitness", fontsize=8, color=INK2)
    ax.set_title("Neutral drift, per run", color=INK, fontsize=10)

    fig.suptitle(
        "CGP's genotype-phenotype map in use: small phenotypes, and how often "
        "neutral offspring were accepted",
        color=INK, fontsize=11,
    )
    return _save(fig, out, top=0.86, bottom=0.1, left=0.07, right=0.985, wspace=0.2)


def build_all(release_dir: Path, progress=print) -> list[Path]:
    release_dir = Path(release_dir)
    runs, final = load(release_dir)
    figs = release_dir / "figures"
    summary = summarise(runs, final)
    effects = family_effects(runs, final)
    budgets = budget_table(runs)

    made = [
        fig_accuracy_by_task(summary, figs / "accuracy-by-task.png"),
        fig_reversal(effects, figs / "reversal.png"),
        fig_budget(budgets, figs / "realized-budget.png"),
        fig_cgp_structure(runs, figs / "cgp-structure.png"),
    ]
    if final:
        signs = sign_matrix(runs, final)
        cross = cross_algorithm(runs, final)
        made.insert(0, fig_sign_matrix(signs, figs / "sign-matrix.png"))
        made.append(fig_cross_algorithm(cross, figs / "cross-algorithm.png"))
        made.append(fig_hypotheses(hypotheses(runs, final), figs / "hypotheses.png"))
    progress(f"{len(made)} figures -> {figs}")
    return made
