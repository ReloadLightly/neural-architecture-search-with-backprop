"""Pictures of the things themselves: the networks, and what they compute.

This repository is about network architectures and two-dimensional decision
boundaries, and for four protocols it showed neither. Every figure in it was a
summary statistic. These are the primary objects.

Nothing is retrained and no search is re-run: every champion is loaded from a
committed run record exactly as validation selected it, and every boundary is
that champion evaluated on a grid spanning the *training* points. The sealed
test split is never read — `bpneat.champions.boundary_field` takes its grid from
`bundle.train` alone, which the poison gate asserts.

Every panel here is a measured result. The five input geometries used to be
drawn here as well; they are the problem setup rather than a finding, so that
figure is gone and `tests/test_readme_figures.py` keeps it gone.

Look: the project's one style, which is the style of
`competitive-coevolution-of-slimes` — white ground, no frame and no border, a
very light grid behind the data, and a left-aligned sentence over each panel
saying what it shows. Every hue, type size and density here comes from
`bpneat.style`; this module defines none of them.

Reuses `bpneat.champions`, which has drawn boundaries and topologies since the
v2 polish pass and was never pointed at the current releases.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bpneat.champions import Champion, boundary_field, topology_layout  # noqa: E402
from bpneat.record import deserialise_genome  # noqa: E402
from bpneat.v3.datasets import make_bundle  # noqa: E402

V4 = ROOT / "results" / "backprop-neat-v4"
V5 = ROOT / "results" / "backprop-neat-v5"
OUT = ROOT / "docs" / "figures"

from bpneat.style import (  # noqa: E402
    ANNOT_SIZE,
    boundary_cmap,
    CLASS_COLOURS,
    GRID,
    INK,
    INK2,
    LABEL_SIZE,
    LEGEND_SIZE,
    NEUTRAL,
    OP_COLOUR,
    panel,
    ramp,
    REFERENCE,
    save,
    style_axes,
    suptitle,
    SURFACE,
    TASK_LABEL,
    TICK_SIZE,
    title,
    TITLE_SIZE,
    WEIGHT_NEG,
    WEIGHT_POS,
    wrap_to,
)

#: Class 0 -> cool, class 1 -> warm, through a near-white midpoint. Two hues
#: with a neutral middle is the diverging rule; the midpoint is the decision
#: boundary, so it has to read as "undecided" rather than as a colour.
BOUNDARY_CMAP = boundary_cmap()


def _load(release: Path) -> tuple[list[dict], dict[str, dict]]:
    runs = [json.loads(p.read_text())
            for p in sorted((release / "raw" / "runs").glob("*.json"))]
    final = {}
    fpath = release / "final-test.json"
    if fpath.exists():
        final = {r["run_id"]: r for r in json.loads(fpath.read_text())["results"]}
    return runs, final


def _median_champion(runs, final, task: str, condition: str) -> Champion | None:
    """The replicate that scored the median — never the best one.

    Showing a best-of-N champion would be the selection error this whole project
    is about. The median replicate is what the condition typically produces.
    """
    items = [r for r in runs if r["task"] == task and r["condition"] == condition]
    if not items:
        return None
    key = (lambda r: final[r["run_id"]]["test_accuracy"]) if final else (
        lambda r: r["metrics"]["validation_accuracy"])
    items.sort(key=key)
    rec = items[(len(items) - 1) // 2]
    g, w = deserialise_genome(rec["champion"])
    return Champion(
        task=task, run_id=rec["run_id"], replicate=rec["replicate"], track="",
        condition=condition, genome=g, weights=w, settle=True,
        dataset_seed=rec["dataset_seed"],
        test_accuracy=final[rec["run_id"]]["test_accuracy"] if final else None,
        n_candidates=len(items), rank=(len(items) - 1) // 2,
    )


def _save(fig, out: Path, **adjust) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = save(fig, out, **adjust)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def boundary_key(fig, y: float = 0.05) -> None:
    """What the two pigments mean, said once per figure, under the panels.

    Thirty-three two-colour panels went out with nothing anywhere stating that
    cold is class 0, warm is class 1, and the near-white middle of the ramp is
    the model declining to decide. A figure that never names its encoding is
    asking the reader to guess it.

    The earlier version measured this band off a page border and sat inside it.
    There is no border any more, so the bar and its caption simply sit a fixed
    distance up from the bottom edge and the caller reserves room for them with
    `bottom=`; only the bar's thickness is in inches, so it is the same weight
    on a tall figure as on a short one.
    """
    height = fig.get_size_inches()[1]
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    bar = fig.add_axes([0.40, y, 0.20, 0.09 / height])
    bar.imshow(gradient, aspect="auto", cmap=BOUNDARY_CMAP, vmin=0, vmax=1)
    panel(bar)
    mid = y + 0.045 / height
    fig.text(0.395, mid, "predicts class 0", ha="right", va="center",
             fontsize=LABEL_SIZE, color=INK2)
    fig.text(0.605, mid, "predicts class 1", ha="left", va="center",
             fontsize=LABEL_SIZE, color=INK2)
    # Wrapped to the figure, not to a character count: this one line was the
    # last artist still pushing a tight-cropped figure past the page.
    caption = wrap_to(
        "the near-white middle of the ramp is the model undecided; the dark "
        "line is the 0.5 contour; dots are training points, coloured by "
        "their true class",
        fig.get_size_inches()[0] - 0.3, ANNOT_SIZE, fig,
    )
    fig.text(0.5, y - 0.19 / height, caption,
             ha="center", va="center", fontsize=ANNOT_SIZE, color=INK2,
             linespacing=1.45)


def _draw_boundary(ax, champ: Champion, show_points: bool = True):
    bundle = make_bundle(champ.task, seed=champ.dataset_seed)
    field = boundary_field(champ, bundle, resolution=200)
    ax.imshow(field["prob"], extent=field["extent"], origin="lower",
              cmap=BOUNDARY_CMAP, vmin=0.0, vmax=1.0, interpolation="bilinear",
              aspect="equal")
    # The 0.5 contour is the decision boundary itself; drawing it means the
    # reader does not have to infer it from a colour gradient.
    ax.contour(field["xx"], field["yy"], field["prob"], levels=[0.5],
               colors=[INK], linewidths=1.1, alpha=0.75)
    if show_points:
        X, y = field["train_X"], field["train_y"]
        for cls, colour in zip((0.0, 1.0), CLASS_COLOURS):
            m = y == cls
            # The white keyline is what keeps two hundred overlapping dots
            # readable as dots rather than as one dark mass.
            ax.scatter(X[m, 0], X[m, 1], s=5.5, c=colour, linewidths=0.4,
                       edgecolors=SURFACE, alpha=0.9, zorder=3)
    panel(ax)


def fig_decision_boundaries(out: Path) -> Path:
    """What each condition's median champion actually computes."""
    runs, final = _load(V4)
    tasks = ("spiral", "checkerboard", "spiral3")
    cols = [
        ("bpneat", "Backprop-NEAT"),
        ("cgp", "CGP"),
        ("cgp_random_matched", "random CGP\n(no selection)"),
        ("fixed_tanh_ha", "fixed net,\nstarved"),
        ("fixed_mixed_matched_bpneat", "fixed net,\nmatched"),
    ]
    fig, axes = plt.subplots(len(tasks), len(cols),
                             figsize=(1.32 * len(cols), 1.46 * len(tasks)),
                             facecolor=SURFACE)
    for ri, task in enumerate(tasks):
        for ci, (cond, label) in enumerate(cols):
            ax = axes[ri, ci]
            champ = _median_champion(runs, final, task, cond)
            if champ is None:
                ax.axis("off")
                continue
            _draw_boundary(ax, champ)
            acc = champ.test_accuracy
            ax.set_xlabel(f"{acc:.3f}" if acc is not None else "",
                          fontsize=LABEL_SIZE, color=INK, labelpad=3)
            if ri == 0:
                title(ax, label)
            if ci == 0:
                ax.set_ylabel(TASK_LABEL[task], fontsize=LABEL_SIZE, color=INK,
                              labelpad=7)
    # The headline is a sentence, flush left like every other title here.
    suptitle(fig,
        "What the search actually built — the median champion's decision boundary\n"
        "Sealed-test accuracy under each panel. Median replicate of 30, never the best.",
        x=0.0, ha="left", color=INK, fontsize=TITLE_SIZE,
    )
    boundary_key(fig)
    return _save(fig, out, top=0.78, bottom=0.17, left=0.055, right=0.99,
                 wspace=0.07, hspace=0.30)


def _draw_network(ax, champ: Champion, heading: str, subtitle: str):
    bundle = make_bundle(champ.task, seed=champ.dataset_seed)
    lay = topology_layout(champ, bundle)
    wmax = max((abs(e["weight"]) for e in lay["edges"]), default=1.0) or 1.0

    for e in lay["edges"]:
        causal = e["causal"]
        ax.plot(
            [e["x0"], e["x1"]], [e["y0"], e["y1"]],
            # A weight's sign is a diverging quantity, so it takes the
            # project's declared diverging anchors. Using the categorical
            # series here meant blue was "tanh" on a node and "positive" on
            # the edge leaving it.
            color=(WEIGHT_POS if e["weight"] >= 0 else WEIGHT_NEG) if causal else GRID,
            linewidth=(0.5 + 2.6 * abs(e["weight"]) / wmax) if causal else 0.6,
            alpha=0.85 if causal else 0.4, zorder=1,
            solid_capstyle="round",
        )
    for n in lay["nodes"]:
        colour = OP_COLOUR.get(n["operator"], NEUTRAL)
        # A node that never reaches the output is drawn hollow: white fill, the
        # grid's own grey for its ring, so it is plainly structure rather than
        # a tenth operator.
        ax.scatter(n["x"], n["y"], s=150 if n["structural"] else 110,
                   c=colour if n["causal"] else SURFACE,
                   edgecolors=INK2 if n["causal"] else GRID,
                   linewidths=1.1, zorder=3,
                   marker="s" if n["structural"] else "o")
    title(ax, heading)
    ax.text(0.5, -0.09, subtitle, transform=ax.transAxes, ha="center",
            fontsize=ANNOT_SIZE, color=INK2)
    ax.set_xlim(-0.6, lay["n_columns"] - 0.4)
    ax.set_ylim(-1.25, 1.25)
    ax.axis("off")
    return lay


def fig_champion_networks(out: Path) -> Path:
    """The evolved graphs, drawn. Solid = causal, ghosted = represented only."""
    runs, final = _load(V4)
    panels = [
        ("spiral", "bpneat", "Backprop-NEAT"),
        ("spiral", "cgp", "CGP"),
        ("spiral", "cgp_random_matched", "random CGP"),
        ("checkerboard", "bpneat", "Backprop-NEAT"),
        ("checkerboard", "cgp", "CGP"),
        ("checkerboard", "cgp_random_matched", "random CGP"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.9), facecolor=SURFACE)
    for ax, (task, cond, label) in zip(axes.ravel(), panels):
        champ = _median_champion(runs, final, task, cond)
        if champ is None:
            ax.axis("off")
            continue
        lay = topology_layout(champ, make_bundle(task, seed=champ.dataset_seed))
        _draw_network(
            ax, champ, f"{label} — {TASK_LABEL[task]}",
            f"{lay['causal_hidden']} causal of {lay['represented_hidden']} hidden units, "
            f"{lay['causal_connections']} of {lay['represented_connections']} edges",
        )

    ops_present = sorted({
        n["operator"]
        for (task, cond, _) in panels
        if (c := _median_champion(runs, final, task, cond)) is not None
        for n in topology_layout(c, make_bundle(task, seed=c.dataset_seed))["nodes"]
        if not n["structural"]
    })
    # Every swatch carries the same ink ring the nodes do: several of the
    # palette's hues are light enough against the page that an unringed swatch
    # at legend size reads as a blank rather than as a colour.
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8,
                          markerfacecolor=OP_COLOUR.get(op, NEUTRAL),
                          markeredgecolor=INK2, markeredgewidth=1.0,
                          color="none", label=op)
               for op in ops_present]
    handles += [
        plt.Line2D([], [], color=WEIGHT_POS, linewidth=2.4, label="positive weight"),
        plt.Line2D([], [], color=WEIGHT_NEG, linewidth=2.4, label="negative weight"),
        plt.Line2D([], [], color=GRID, linewidth=1.2,
                   label="represented but never reaches the output"),
    ]
    # Columns chosen by measurement, not by eye: at six columns this legend
    # was 7.7 inches wide on a 6.6-inch figure, and a tight crop grows the page
    # to fit a legend rather than wrapping it.
    for ncol in range(6, 1, -1):
        legend = fig.legend(handles=handles, fontsize=LEGEND_SIZE, frameon=False,
                            ncol=ncol, loc="lower center")
        fig.canvas.draw()
        if legend.get_window_extent().width / fig.dpi <= fig.get_size_inches()[0] - 0.2:
            break
        legend.remove()
    suptitle(fig,
        "The networks themselves. Edge width is |weight|; a hollow node and a pale\n"
        "edge are structure the genome carries that never reaches the output.",
        x=0.0, ha="left", color=INK, fontsize=TITLE_SIZE,
    )
    # The networks draw their own axes, whose tick labels sit outside the
    # panel; the margins have to hold them or the tight crop grows the page.
    return _save(fig, out, top=0.80, bottom=0.17, left=0.10, right=0.90,
                 wspace=0.05, hspace=0.30)


def fig_mechanism_boundaries(out: Path) -> Path:
    """v5: what each NEAT mechanism does to the function, not just the score."""
    if not (V5 / "raw" / "runs").is_dir():
        return None
    runs, final = _load(V5)
    cols = [
        ("neat_reference", "reference"),
        ("neat_complexify", "+ mutation rate"),
        ("neat_no_penalty", "− penalty"),
        ("neat_complexify_no_penalty", "both"),
        ("neat_deep_narrow", "deep & narrow"),
        ("fixed_mixed_matched", "fixed net"),
    ]
    tasks = ("spiral", "checkerboard", "spiral3")
    fig, axes = plt.subplots(len(tasks), len(cols),
                             figsize=(1.1 * len(cols), 1.34 * len(tasks)),
                             facecolor=SURFACE)
    for ri, task in enumerate(tasks):
        for ci, (cond, label) in enumerate(cols):
            ax = axes[ri, ci]
            champ = _median_champion(runs, final, task, cond)
            if champ is None:
                ax.axis("off")
                continue
            _draw_boundary(ax, champ)
            acc = champ.test_accuracy
            if acc is None:
                acc = next(r["metrics"]["validation_accuracy"] for r in runs
                           if r["run_id"] == champ.run_id)
            ax.set_xlabel(f"{acc:.3f}", fontsize=LABEL_SIZE, color=INK, labelpad=3)
            if ri == 0:
                title(ax, label)
            if ci == 0:
                ax.set_ylabel(TASK_LABEL[task], fontsize=LABEL_SIZE, color=INK,
                              labelpad=7)
    suptitle(fig,
        "What each NEAT mechanism changes about the function it finds\n"
        "Median champion of 30 replicates; accuracy under each panel.",
        x=0.0, ha="left", color=INK, fontsize=TITLE_SIZE,
    )
    boundary_key(fig)
    return _save(fig, out, top=0.78, bottom=0.17, left=0.09, right=0.96,
                 wspace=0.07, hspace=0.30)


#: The qualitative reading in the published description (Neuroevolution §10.1,
#: recorded in docs/reference-targets.md): XOR is said to rely on abs and relu,
#: circles on sine, square and gaussian. It is a hypothesis to test, never a
#: criterion for selecting which runs to report.
PREDICTED = {"xor": ("abs", "relu"), "circle": ("sin", "square", "gaussian")}

#: The reference search condition in each release, under its own name.
REFERENCE_OF = {"v3": "backprop_neat", "v4": "bpneat", "v5": "neat_reference"}


def fig_operator_usage(out: Path) -> Path:
    """Which operators the search actually selects, across three protocols.

    Causal operator fractions — counted over the operators that reached the
    output, not the ones the genome carries. Three protocols with disjoint seeds
    and separate code packages, so the spread across the three dots is a
    replication check rather than decoration.
    """
    import csv

    data: dict[str, dict[str, dict[str, float]]] = {}
    for tag, cond in REFERENCE_OF.items():
        path = ROOT / "results" / f"backprop-neat-{tag}" / "operator-usage.csv"
        if not path.exists():
            continue
        with open(path) as fh:
            for r in csv.DictReader(fh):
                if r["condition"] != cond:
                    continue
                data.setdefault(r["task"], {}).setdefault(tag, {})[r["operator"]] = float(
                    r["fraction"]
                )

    tasks = [t for t in ("xor", "circle", "spiral", "checkerboard", "spiral3") if t in data]
    # Not sharex: each geometry concentrates its usage differently, and one
    # shared 0..0.88 range left three quarters of four panels empty.
    fig, axes = plt.subplots(len(tasks), 1, figsize=(6.6, 1.65 * len(tasks)),
                             facecolor=SURFACE)
    # Three protocols measuring the same reference search, so one family at
    # three values — not three unrelated hues, which previously drew the
    # algorithm under study in the fixed control's colour.
    _proto = ramp("search_primary", 3)
    style = {"v3": (_proto[0], "o"), "v4": (_proto[1], "s"), "v5": (_proto[2], "^")}

    for ax, task in zip(np.atleast_1d(axes).ravel(), tasks):
        # A light grid along the value axis, behind the data: the shared
        # recessive frame, with the left spine dropped because the operator
        # names are the axis.
        style_axes(ax, grid_axis="x")
        ax.spines["left"].set_visible(False)
        per = data[task]
        ops = sorted(
            {o for d in per.values() for o in d},
            key=lambda o: -np.mean([d.get(o, 0.0) for d in per.values()]),
        )
        y = np.arange(len(ops))[::-1]
        for yi, op in zip(y, ops):
            vals = [d.get(op, 0.0) for d in per.values()]
            # The spread between protocols is the replication check, i.e. the
            # figure's actual claim. In GRID it was the colour of the gridlines
            # it crossed; in the reference ink it reads as a measurement.
            ax.plot([min(vals), max(vals)], [yi, yi], color=REFERENCE, linewidth=4.5,
                    alpha=0.45, solid_capstyle="round", zorder=1)
            for tag, d in per.items():
                colour, marker = style[tag]
                ax.scatter(d.get(op, 0.0), yi, s=58, color=colour, marker=marker,
                           edgecolors=SURFACE, linewidths=1.3, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [f"{op} *" if op in PREDICTED.get(task, ()) else op for op in ops],
            fontsize=TICK_SIZE,
        )
        # The asterisk marks an operator the published description predicts for
        # this geometry, which is the figure's hypothesis test. Give those rows
        # the primary ink so the test is visible without reading the caption.
        for lab, op in zip(ax.get_yticklabels(), ops):
            predicted = op in PREDICTED.get(task, ())
            lab.set_color(INK if predicted else INK2)
            lab.set_fontweight("semibold" if predicted else "normal")
        title(ax, TASK_LABEL[task])
        # Pinned per panel to where the data are: a shared 0..0.88 left three
        # quarters of four panels empty and compressed every difference that
        # matters into the leftmost eighth.
        top = max(max(d.values(), default=0.0) for d in per.values())
        ax.set_xlim(-0.02, min(0.9, top * 1.22 + 0.04))

    handles = [plt.Line2D([], [], marker=m, linestyle="", markersize=8, color=c,
                          markeredgecolor=SURFACE, label=f"protocol {t}")
               for t, (c, m) in style.items()]
    fig.legend(handles=handles, fontsize=LEGEND_SIZE, frameon=False, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.005))
    suptitle(fig,
        "Which operators the search selects — and what the published description "
        "predicts (*)\n"
        "Fraction of causally active hidden nodes. Three protocols, disjoint seeds, "
        "separate code.",
        x=0.0, ha="left", color=INK, fontsize=TITLE_SIZE,
    )
    # Stacked, each panel carries its own x axis, so the gap between rows has
    # to clear a tick row and the next panel's heading.
    return _save(fig, out, top=0.895, bottom=0.07, left=0.13, right=0.99,
                 hspace=0.62)


def main() -> int:
    if not (V4 / "summary.csv").exists():
        print("v4 release missing", file=sys.stderr)
        return 1
    fig_decision_boundaries(OUT / "decision-boundaries.png")
    fig_champion_networks(OUT / "champion-networks.png")
    fig_mechanism_boundaries(OUT / "mechanism-boundaries.png")
    fig_operator_usage(OUT / "operator-usage.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
