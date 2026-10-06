"""The figures this polish pass added, all rendered from the committed release.

Three drawings, none of which retrains or re-runs anything:

* **Decision boundaries** — the champion genome and its learned weights, loaded
  from the run record exactly as the search left them, evaluated on a grid over
  the input range. The training points are drawn on top; the sealed test split
  is never read.
* **Champion topologies** — nodes coloured by operator and laid out by BFS
  depth, with the causal subgraph drawn solid and merely represented structure
  ghosted. This is the release's H5 finding made visible: the gap between what
  a network contains and what reaches its output.
* **Paired sealed-test loss, track B** — the release README's headline table as
  a forest plot, in the style of the existing paired-effects figure.

**Display choice, not a selection claim.** For each task the champion drawn is
the track-B ``backprop_neat`` replicate with the median sealed-test accuracy —
sorted by accuracy then replicate id, the lower of the two central replicates,
since ten replicates have no single middle. The rule is fixed in advance and
stated in every caption. No replicate was chosen for how it looks.

Array computation is kept separate from rendering: :func:`boundary_field` and
:func:`topology_layout` return plain arrays and dictionaries and import no
plotting code, which is what lets the sealed-test poison gate test them
directly.

Every hue, type size and density below comes from :mod:`bpneat.style`, which is
the `competitive-coevolution-of-slimes` look ported whole: white ground, no
frame, a left-aligned sentence for a title, marks keylined in the page colour.
Nothing here chooses a colour of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .datasets import DatasetBundle, make_bundle
from .genome import (
    _MUL_W,
    _PROD,
    _SUM,
    BIAS,
    IN_X,
    IN_Y,
    N_STRUCTURAL,
    OP_NAMES,
    OUT,
    Genome,
    causal_subgraph,
    forward,
    predict_logits,
)
from .record import deserialise_genome

TASKS = ("xor", "circle", "spiral")
REFERENCE_CONDITION = "backprop_neat"

STRUCTURAL_LABEL = {BIAS: "bias", IN_X: "x", IN_Y: "y", OUT: "out"}


# --------------------------------------------------------------------------
# Selection — deterministic, stated, and independent of how a drawing looks
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Champion:
    """One selected run, with everything a drawing needs and nothing more."""

    task: str
    run_id: str
    replicate: int
    track: str
    condition: str
    genome: Genome
    weights: np.ndarray
    settle: bool
    dataset_seed: int
    test_accuracy: float | None
    n_candidates: int
    rank: int

    @property
    def caption(self) -> str:
        acc = "n/a" if self.test_accuracy is None else f"{self.test_accuracy:.3f}"
        return (
            f"replicate {self.replicate} of {self.n_candidates}, "
            f"sealed-test accuracy {acc}"
        )


def _median_index(n: int) -> int:
    """The lower of the two central positions, so an even count is decidable."""
    return (n - 1) // 2


def select_champions(
    release_dir: Path,
    track_dir: str = "track-b",
    condition: str = REFERENCE_CONDITION,
) -> dict[str, Champion]:
    """Per task, the median-sealed-test-accuracy replicate of ``condition``.

    Ties break on the lower replicate id. Reads only committed records.
    """
    from .analysis import load_final_test, load_runs

    root = Path(release_dir)
    track = root / track_dir
    runs = [r for r in load_runs(track) if r["condition"] == condition]
    final = load_final_test(track)

    chosen: dict[str, Champion] = {}
    for task in TASKS:
        items = [r for r in runs if r["task"] == task]
        if not items:
            continue
        # Sort on (sealed-test accuracy, replicate). A run without a test
        # metric sorts last rather than silently taking the middle.
        def key(rec):
            t = final.get(rec["run_id"])
            acc = -np.inf if t is None else t["test_accuracy"]
            return (acc, rec["replicate"])

        items = sorted(items, key=key)
        rank = _median_index(len(items))
        rec = items[rank]
        g, w = deserialise_genome(rec["champion"])
        t = final.get(rec["run_id"])
        chosen[task] = Champion(
            task=task,
            run_id=rec["run_id"],
            replicate=rec["replicate"],
            track=rec["track"],
            condition=rec["condition"],
            genome=g,
            weights=w,
            settle=rec["config"]["propagation"] == "settled",
            dataset_seed=rec["dataset_seed"],
            test_accuracy=None if t is None else float(t["test_accuracy"]),
            n_candidates=len(items),
            rank=rank,
        )
    return chosen


def champion_bundle(champion: Champion) -> DatasetBundle:
    """The dataset the champion was searched under. Only ``train`` is ever read."""
    return make_bundle(champion.task, seed=champion.dataset_seed)


# --------------------------------------------------------------------------
# Computation — no plotting imports below this line until the render section
# --------------------------------------------------------------------------


def boundary_field(
    champion: Champion,
    bundle: DatasetBundle,
    resolution: int = 220,
    margin: float = 0.06,
    square: bool = True,
) -> dict:
    """Champion output over a grid spanning the training points.

    The grid range comes from ``bundle.train`` alone. ``bundle.test`` is not
    read here or anywhere downstream, which the poison-gate test asserts.
    """
    X = bundle.train.X
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    lo = lo - margin * span
    hi = hi + margin * span
    if square:
        # A square window keeps the geometry undistorted under an equal aspect
        # and gives every panel the same shape, so the titles align.
        centre = (lo + hi) / 2.0
        half = float(np.max(hi - lo)) / 2.0
        lo, hi = centre - half, centre + half

    xs = np.linspace(lo[0], hi[0], resolution)
    ys = np.linspace(lo[1], hi[1], resolution)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)

    logits = predict_logits(champion.genome, grid, champion.weights, settle=champion.settle)
    prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    return {
        "xx": xx,
        "yy": yy,
        "prob": prob.reshape(xx.shape),
        "extent": (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])),
        "train_X": X,
        "train_y": bundle.train.y,
    }


def causal_elements(
    g: Genome, X: np.ndarray, weights: np.ndarray, settle: bool
) -> tuple[set[int], set[int]]:
    """The connections and nodes that actually reached the output.

    ``genome.causal_subgraph`` returns the counts this walk implies but not the
    sets a drawing needs, and ``genome.py`` is frozen to the release
    fingerprint. So the same backward walk over the executed tape is repeated
    here, and :func:`causal_counts_agree` checks the two never diverge.
    """
    tape = forward(g, X, weights, settle=settle)
    live_vars = {tape.out_var}
    live_conns: set[int] = set()

    for entry in reversed(tape.entries):
        if entry[1] not in live_vars:
            continue
        kind = entry[0]
        if kind == _MUL_W:
            live_conns.add(entry[3])
            live_vars.add(entry[2])
        elif kind in (_SUM, _PROD):
            live_vars.update(entry[2])
        else:  # _UNARY_OP, _SQUARE, _CLAMP
            live_vars.add(entry[2])

    nodes = {g.src[ci] for ci in live_conns} | {g.dst[ci] for ci in live_conns}
    return live_conns, nodes


def causal_counts_agree(
    g: Genome, X: np.ndarray, weights: np.ndarray, settle: bool
) -> bool:
    """Cross-check the local walk against the frozen implementation."""
    conns, nodes = causal_elements(g, X, weights, settle)
    ref = causal_subgraph(g, X, weights, settle=settle)
    hidden = {i for i in nodes if i >= N_STRUCTURAL}
    return (
        len(conns) == ref["causal_connections"]
        and len(nodes) == ref["causal_nodes"]
        and len(hidden) == ref["causal_hidden_nodes"]
    )


def _bfs_depth(g: Genome) -> dict[int, int]:
    """Shortest distance from the input layer along enabled connections."""
    outgoing: dict[int, list[int]] = {}
    for ci in range(g.n_connections):
        if g.active[ci]:
            outgoing.setdefault(g.src[ci], []).append(g.dst[ci])

    depth = {BIAS: 0, IN_X: 0, IN_Y: 0}
    frontier = [BIAS, IN_X, IN_Y]
    while frontier:
        nxt = []
        for node in frontier:
            for dst in outgoing.get(node, ()):
                if dst not in depth:
                    depth[dst] = depth[node] + 1
                    nxt.append(dst)
        frontier = nxt
    return depth


def topology_layout(
    champion: Champion, bundle: DatasetBundle, max_points: int = 64
) -> dict:
    """Node positions, operators, and which structure is causal.

    Only ``bundle.train`` is read — the causal walk needs *some* input to
    execute the graph on, and it is the training split by construction.
    """
    g = champion.genome
    X = bundle.train.X[:max_points]
    live_conns, live_nodes = causal_elements(g, X, champion.weights, champion.settle)

    depth = _bfs_depth(g)
    reachable = [d for i, d in depth.items() if i != OUT]
    last_column = (max(reachable) if reachable else 0) + 1
    # The output always sits in the rightmost column so the drawing reads
    # left to right, whatever path length actually reaches it.
    column = {i: min(depth.get(i, last_column - 1), last_column - 1) for i in range(g.n_nodes)}
    column[BIAS] = column[IN_X] = column[IN_Y] = 0
    column[OUT] = last_column

    by_column: dict[int, list[int]] = {}
    for node in range(g.n_nodes):
        by_column.setdefault(column[node], []).append(node)

    pos: dict[int, tuple[float, float]] = {}
    for col, members in by_column.items():
        members.sort()
        n = len(members)
        for k, node in enumerate(members):
            y = 0.0 if n == 1 else 1.0 - 2.0 * k / (n - 1)
            pos[node] = (float(col), float(y))

    nodes = []
    for i in range(g.n_nodes):
        nodes.append(
            {
                "id": i,
                "x": pos[i][0],
                "y": pos[i][1],
                "operator": OP_NAMES[g.ops[i]],
                "label": STRUCTURAL_LABEL.get(i, OP_NAMES[g.ops[i]]),
                "structural": i < N_STRUCTURAL,
                "causal": i in live_nodes or i == OUT,
            }
        )

    edges = []
    for ci in range(g.n_connections):
        if not g.active[ci]:
            continue
        s, d = g.src[ci], g.dst[ci]
        edges.append(
            {
                "connection": ci,
                "src": s,
                "dst": d,
                "x0": pos[s][0],
                "y0": pos[s][1],
                "x1": pos[d][0],
                "y1": pos[d][1],
                "weight": float(champion.weights[ci]),
                "causal": ci in live_conns,
            }
        )

    hidden_causal = {i for i in live_nodes if i >= N_STRUCTURAL}
    return {
        "nodes": nodes,
        "edges": edges,
        "n_columns": last_column + 1,
        "represented_hidden": g.n_nodes - N_STRUCTURAL,
        "causal_hidden": len(hidden_causal),
        "represented_connections": g.n_enabled,
        "causal_connections": len(live_conns),
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _plotting():
    """Deferred import, so the computation above stays plot-free.

    Returns the style module itself rather than a dictionary of names copied
    out of it. Two reasons: at the use site ``s.INK`` says where the ink came
    from, and there is no second place in this file where a hue, a type size or
    a density could be introduced. Previously these names were re-exported
    through ``figures.py``'s private helpers, so this module's look depended on
    another figure module's internals; now both read the same source.
    """
    import matplotlib

    # A figure here is written to a file and never shown, and the release tests
    # render it on machines with no display.
    matplotlib.use("Agg")

    from . import style

    return style


def champion_boundaries(release_dir: Path, out: Path, track_dir: str = "track-b") -> Path | None:
    import matplotlib.colors as mcolors

    s = _plotting()
    chosen = select_champions(release_dir, track_dir)
    if not chosen:
        return None

    # The field is washed out toward the page and the points keep the saturated
    # class colours, so a class-1 point stays readable where the model also
    # predicts class 1. The middle of the ramp stays near-neutral because the
    # midpoint *is* the decision boundary: it has to read as undecided rather
    # than as a third class.
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "bpneat-boundary",
        [s.pale(s.BOUNDARY_COLOURS[0]), s.BOUNDARY_COLOURS[1],
         s.pale(s.BOUNDARY_COLOURS[2])],
    )
    marker = {0: "o", 1: "^"}
    tasks = [t for t in TASKS if t in chosen]
    # No grid behind these panels: they are images of the input plane, and a
    # ruler drawn across a continuous field measures nothing.
    # Panels stack downwards: side by side, three of them made a figure wider
    # than the page it is read on, which is what makes a label unreadable.
    fig, axes = s.figure(len(tasks), 1, figsize=(6.6, 3.1 * len(tasks)),
                         grid_axis=None)
    for ax, task in zip(np.atleast_1d(axes).ravel(), tasks):
        champ = chosen[task]
        field = boundary_field(champ, champion_bundle(champ))
        ax.imshow(
            field["prob"], extent=field["extent"], origin="lower", cmap=cmap,
            vmin=0.0, vmax=1.0, aspect="auto", interpolation="bilinear", zorder=0,
        )
        ax.contour(
            field["xx"], field["yy"], field["prob"],
            levels=[0.5], colors=[s.INK], linewidths=1.3, zorder=1,
        )
        X, y = field["train_X"], field["train_y"]
        for cls, colour in enumerate(s.CLASS_COLOURS):
            m = y == cls
            # `style.dot` gives every mark the white keyline, which is what
            # stops 200 overlapping points becoming one blob — and separates a
            # point from the field of its own colour underneath it.
            s.dot(ax, X[m, 0], X[m, 1], colour, marker=marker[cls], size=17,
                  label=f"class {cls}")
        acc = "n/a" if champ.test_accuracy is None else f"{champ.test_accuracy:.3f}"
        s.title(
            ax,
            f"{s.TASK_LABEL[task]}: what the champion learned\n"
            f"replicate {champ.replicate}, sealed-test accuracy {acc}",
        )
        ax.set_aspect("equal")
    # No box round the legend: the field under it is washed out, so a frame
    # would be the only hard edge in the panel.
    np.atleast_1d(axes).ravel()[0].legend(
        frameon=False, fontsize=s.LEGEND_SIZE, labelcolor=s.INK2, loc="upper left",
    )
    s.suptitle(fig,
        "Track B Backprop-NEAT champions: decision boundary over the training points",
        x=0.0, ha="left", color=s.INK, fontsize=s.TITLE_SIZE, y=0.99,
    )
    fig.text(
        0.0, 0.02,
        "Per task the replicate with the median sealed-test accuracy — a display "
        "choice fixed in advance, not a selection claim. The sealed test split is "
        "never read; the points are the training split.",
        ha="left", va="bottom", color=s.INK2, fontsize=s.ANNOT_SIZE,
    )
    # The note below the panels needs room reserved for it; `savefig.bbox`
    # crops to the ink, so these are proportions rather than padding.
    return s.save(fig, Path(out) / "champion-boundaries.png",
                  bottom=0.11, top=0.86, wspace=0.16)


def champion_topologies(release_dir: Path, out: Path, track_dir: str = "track-b") -> Path | None:
    s = _plotting()
    chosen = select_champions(release_dir, track_dir)
    if not chosen:
        return None

    tasks = [t for t in TASKS if t in chosen]
    layouts = {t: topology_layout(chosen[t], champion_bundle(chosen[t])) for t in tasks}
    operators = sorted(
        {n["operator"] for lay in layouts.values() for n in lay["nodes"] if not n["structural"]}
    )
    # Operators have fixed hues shared with every other figure; cycling a
    # six-slot series over them made the same operator change colour.
    palette = {op: s.OP_COLOUR[op] for op in operators}

    # A graph drawing has no axes to rule, so no grid.
    fig, axes = s.figure(len(tasks), 1, figsize=(6.6, 2.9 * len(tasks)),
                         grid_axis=None)
    for ax, task in zip(np.atleast_1d(axes).ravel(), tasks):
        lay = layouts[task]

        for e in lay["edges"]:
            if e["causal"]:
                width = 0.6 + 2.2 * min(abs(e["weight"]), 3.0) / 3.0
                ax.plot(
                    [e["x0"], e["x1"]], [e["y0"], e["y1"]],
                    color=s.INK2, linewidth=width, alpha=0.8,
                    solid_capstyle="round", zorder=1,
                )
            else:
                # Dead structure is a status, not a series, so it takes the
                # reserved neutral: present, legible, and plainly not carrying
                # signal.
                ax.plot(
                    [e["x0"], e["x1"]], [e["y0"], e["y1"]],
                    color=s.NEUTRAL, linewidth=1.0, alpha=0.95,
                    linestyle=(0, (2, 2.5)), zorder=0,
                )

        for n in lay["nodes"]:
            causal = n["causal"]
            colour = s.INK2 if n["structural"] else palette.get(n["operator"], s.INK2)
            # A causal node is filled in its operator's hue; a dead one is
            # filled with the page and keeps only a dashed ring of it, so the
            # two read apart at a glance without a second encoding.
            ax.scatter(
                n["x"], n["y"], s=190 if causal else 130,
                c=[colour] if causal else [s.SURFACE],
                edgecolors=colour, linewidths=1.5,
                linestyle="solid" if causal else "dashed",
                zorder=3,
            )
            # Labels sit under the node: "gaussian" and "sigmoid" do not fit
            # inside a marker, and a label inset on a filled mark is unreadable
            # at this size.
            ax.text(
                n["x"], n["y"] - 0.19, n["label"],
                ha="center", va="top", zorder=4, fontsize=s.ANNOT_SIZE,
                color=s.INK if causal else s.INK2,
                style="normal" if causal else "italic",
            )

        s.title(
            ax,
            f"{s.TASK_LABEL[task]}: {lay['causal_hidden']} of "
            f"{lay['represented_hidden']} hidden nodes compute, "
            f"{lay['causal_connections']} of {lay['represented_connections']} "
            f"connections",
        )
        ax.set_xlim(-0.6, lay["n_columns"] - 0.4)
        ax.set_ylim(-1.55, 1.3)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)

    s.suptitle(fig,
        "Track B Backprop-NEAT champion topologies: what is represented against "
        "what computes",
        x=0.0, ha="left", color=s.INK, fontsize=s.TITLE_SIZE, y=0.99,
    )
    fig.text(
        0.0, 0.02,
        "Solid node and line: reaches the output on the executed trace. Dashed ring "
        "and no fill: represented but causally dead.\n"
        "Columns are BFS depth from the inputs, the output is forced rightmost, "
        "line width is |weight|.",
        ha="left", va="bottom", color=s.INK2, fontsize=s.ANNOT_SIZE,
    )
    return s.save(fig, Path(out) / "champion-topologies.png",
                  bottom=0.11, top=0.86, wspace=0.16)


def paired_test_loss(release_dir: Path, out: Path, track_dir: str = "track-b") -> Path | None:
    """The release README's headline paired table, drawn."""
    from .analysis import load_final_test, load_runs, paired_effects
    from .figures import LABEL

    s = _plotting()
    track = Path(release_dir) / track_dir
    runs = load_runs(track)
    final = load_final_test(track)
    if not runs or not final:
        return None
    rows = paired_effects(runs, final, "test_loss")
    if not rows:
        return None

    track_name = (runs[0].get("track") or track_dir[-1]).upper()
    tasks = [t for t in TASKS if any(r["task"] == t for r in rows)]
    # The grid runs along the measured axis only; a horizontal rule between
    # named rows would only box them in.
    fig, axes = s.figure(len(tasks), 1, figsize=(6.6, 2.4 * len(tasks)),
                         grid_axis="x")
    for ax, task in zip(np.atleast_1d(axes).ravel(), tasks):
        items = sorted(
            [e for e in rows if e["task"] == task], key=lambda e: e["mean_difference"]
        )
        y = np.arange(len(items))
        lo_min = min(e["ci95_low"] for e in items)
        hi_max = max(e["ci95_high"] for e in items)
        width = max(hi_max - lo_min, 1e-9)
        pad = 0.035 * width
        for yi, e in zip(y, items):
            lo, hi = e["ci95_low"], e["ci95_high"]
            # Loss: negative favours Backprop-NEAT, so the colours carry the
            # opposite sense to the accuracy version of this figure.
            colour = s.SEARCH_PRIMARY if e["mean_difference"] < 0 else s.CONTROL_MATCHED
            ax.plot([lo, hi], [yi, yi], color=colour, linewidth=2.4, solid_capstyle="round")
            # The point estimate keeps its white keyline, so it stays visible
            # where it sits near an interval end.
            s.dot(ax, e["mean_difference"], yi, colour, size=46)
            ax.text(
                hi + pad, yi, f"{e['mean_difference']:+.3f}  {e['wins']}/{e['n_pairs']}",
                va="center", fontsize=s.ANNOT_SIZE, color=s.INK,
            )
        # Zero is the reference every interval is read against.
        s.vparity(ax, 0.0)
        ax.set_yticks(y)
        ax.set_yticklabels(["vs " + LABEL[e["comparison"].split(" - ")[1]] for e in items])
        s.title(ax, f"{s.TASK_LABEL[task]}: Backprop-NEAT against each control")
        # The annotations sit to the right of each interval, so the room they
        # need is reserved rather than left to autoscaling.
        ax.set_xlim(lo_min - 0.10 * width, hi_max + 0.62 * width)
        ax.set_ylim(-0.7, len(items) - 0.3)
    np.atleast_1d(axes).ravel()[0].set_xlabel(
        "paired difference in sealed-test loss\n"
        "(Backprop-NEAT − control; negative favours Backprop-NEAT)"
    )
    s.suptitle(fig,
        f"Track {track_name}: paired within-replicate sealed-test loss, "
        "95% bootstrap interval, wins out of pairs",
        x=0.0, ha="left", color=s.INK, fontsize=s.TITLE_SIZE, y=0.99,
    )
    return s.save(fig, Path(out) / f"paired-test-loss-track-{track_name.lower()}.png",
                  top=0.84, wspace=0.3)


def build_all(root: Path, track_dir: str = "track-b", progress=print) -> list[Path]:
    """Render every figure this module owns, into ``<root>/figures``."""
    root = Path(root)
    out = root / "figures"
    written = [
        champion_boundaries(root, out, track_dir),
        champion_topologies(root, out, track_dir),
        paired_test_loss(root, out, track_dir),
    ]
    written = [w for w in written if w]
    for path in written:
        progress(f"wrote {path}")
    return written
