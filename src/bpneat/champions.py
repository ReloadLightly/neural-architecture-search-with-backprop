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
    """Deferred import so the computation above stays plot-free."""
    from .figures import GRID, INK, INK_2, SURFACE, TASK_LABEL, _fig, _save, _style
    from .style import (
        BOUNDARY_COLOURS,
        CLASS_COLOURS,
        CONTROL_MATCHED,
        OP_COLOUR,
        SEARCH_PRIMARY,
    )

    return {
        "GRID": GRID,
        "INK": INK,
        "INK_2": INK_2,
        "BOUNDARY_COLOURS": BOUNDARY_COLOURS,
        "CLASS_COLOURS": CLASS_COLOURS,
        "CONTROL_MATCHED": CONTROL_MATCHED,
        "OP_COLOUR": OP_COLOUR,
        "SEARCH_PRIMARY": SEARCH_PRIMARY,
        "SURFACE": SURFACE,
        "TASK_LABEL": TASK_LABEL,
        "_fig": _fig,
        "_save": _save,
        "_style": _style,
    }


def _pale(colour: str, amount: float = 0.74):
    """Toward white, so the field never competes with the data points on top."""
    import matplotlib.colors as mcolors

    rgb = np.array(mcolors.to_rgb(colour))
    return tuple(rgb + (1.0 - rgb) * amount)


def champion_boundaries(release_dir: Path, out: Path, track_dir: str = "track-b") -> Path | None:
    import matplotlib.colors as mcolors

    s = _plotting()
    chosen = select_champions(release_dir, track_dir)
    if not chosen:
        return None

    # The field is washed out and the points are saturated with a dark edge, so
    # a class-1 point stays readable where the model also predicts class 1.
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "bpneat-boundary",
        [_pale(s["BOUNDARY_COLOURS"][0]), "#ffffff", _pale(s["BOUNDARY_COLOURS"][2])],
    )
    marker = {0: "o", 1: "^"}
    tasks = [t for t in TASKS if t in chosen]
    fig, axes = s["_fig"](1, len(tasks), figsize=(3.7 * len(tasks), 4.6))
    for ax, task in zip(np.atleast_1d(axes).ravel(), tasks):
        champ = chosen[task]
        field = boundary_field(champ, champion_bundle(champ))
        ax.imshow(
            field["prob"], extent=field["extent"], origin="lower", cmap=cmap,
            vmin=0.0, vmax=1.0, aspect="auto", interpolation="bilinear", zorder=0,
        )
        ax.contour(
            field["xx"], field["yy"], field["prob"],
            levels=[0.5], colors=[s["INK"]], linewidths=1.3, zorder=1,
        )
        X, y = field["train_X"], field["train_y"]
        for cls, colour in enumerate(s["CLASS_COLOURS"]):
            m = y == cls
            ax.scatter(
                X[m, 0], X[m, 1], s=17, c=colour, marker=marker[cls],
                linewidths=0.5, edgecolors=s["INK"], zorder=2, label=f"class {cls}",
            )
        acc = "n/a" if champ.test_accuracy is None else f"{champ.test_accuracy:.3f}"
        ax.set_title(
            f"{s['TASK_LABEL'][task]}\nreplicate {champ.replicate} · sealed-test {acc}",
            color=s["INK"], fontsize=9.5,
        )
        ax.set_aspect("equal")
        ax.grid(False)
    np.atleast_1d(axes).ravel()[0].legend(
        frameon=True, facecolor=s["SURFACE"], edgecolor=s["GRID"],
        fontsize=7.5, labelcolor=s["INK_2"], loc="upper left",
    )
    fig.suptitle(
        "Track B Backprop-NEAT champions: decision boundary over the training points",
        color=s["INK"], fontsize=11, y=0.98,
    )
    fig.text(
        0.5, 0.028,
        "Per task the replicate with the median sealed-test accuracy — a display "
        "choice fixed in advance, not a selection claim. The sealed test split is "
        "never read; the points are the training split.",
        ha="center", color=s["INK_2"], fontsize=8,
    )
    return s["_save"](fig, Path(out) / "champion-boundaries.png", bottom=0.11)


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
    palette = {op: s["OP_COLOUR"][op] for op in operators}

    fig, axes = s["_fig"](1, len(tasks), figsize=(3.9 * len(tasks), 4.8))
    for ax, task in zip(np.atleast_1d(axes).ravel(), tasks):
        lay = layouts[task]

        for e in lay["edges"]:
            if e["causal"]:
                width = 0.6 + 2.2 * min(abs(e["weight"]), 3.0) / 3.0
                ax.plot(
                    [e["x0"], e["x1"]], [e["y0"], e["y1"]],
                    color=s["INK_2"], linewidth=width, alpha=0.8,
                    solid_capstyle="round", zorder=1,
                )
            else:
                ax.plot(
                    [e["x0"], e["x1"]], [e["y0"], e["y1"]],
                    color="#b9b7b2", linewidth=1.0, alpha=0.95,
                    linestyle=(0, (2, 2.5)), zorder=0,
                )

        for n in lay["nodes"]:
            causal = n["causal"]
            colour = s["INK_2"] if n["structural"] else palette.get(n["operator"], s["INK_2"])
            ax.scatter(
                n["x"], n["y"], s=190 if causal else 130,
                c=[colour] if causal else [s["SURFACE"]],
                edgecolors=colour, linewidths=1.5,
                linestyle="solid" if causal else "dashed",
                zorder=3,
            )
            # Labels sit under the node: "gaussian" and "sigmoid" do not fit
            # inside a marker, and a pale operator colour makes an inset label
            # unreadable.
            ax.text(
                n["x"], n["y"] - 0.19, n["label"],
                ha="center", va="top", zorder=4, fontsize=7,
                color=s["INK"] if causal else s["INK_2"],
                style="normal" if causal else "italic",
            )

        ax.set_title(
            f"{s['TASK_LABEL'][task]}\n{lay['causal_hidden']} of "
            f"{lay['represented_hidden']} hidden nodes causal · "
            f"{lay['causal_connections']} of {lay['represented_connections']} "
            f"connections",
            color=s["INK"], fontsize=9.5,
        )
        ax.set_xlim(-0.6, lay["n_columns"] - 0.4)
        ax.set_ylim(-1.55, 1.3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)

    fig.suptitle(
        "Track B Backprop-NEAT champion topologies: what is represented against "
        "what computes",
        color=s["INK"], fontsize=11, y=0.98,
    )
    fig.text(
        0.5, 0.028,
        "Solid node and line: reaches the output on the executed trace. Dashed and "
        "hollow: represented but causally dead.\n"
        "Columns are BFS depth from the inputs, the output is forced rightmost, "
        "line width is |weight|.",
        ha="center", color=s["INK_2"], fontsize=8,
    )
    return s["_save"](fig, Path(out) / "champion-topologies.png", bottom=0.11)


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
    fig, axes = s["_fig"](1, len(tasks), figsize=(3.9 * len(tasks), 4.0))
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
            colour = (s["SEARCH_PRIMARY"] if e["mean_difference"] < 0
                      else s["CONTROL_MATCHED"])
            ax.plot([lo, hi], [yi, yi], color=colour, linewidth=2.4, solid_capstyle="round")
            ax.plot(
                e["mean_difference"], yi, "o", color=colour, markersize=8,
                markeredgecolor=s["SURFACE"], markeredgewidth=1.6,
            )
            ax.text(
                hi + pad, yi, f"{e['mean_difference']:+.3f}  {e['wins']}/{e['n_pairs']}",
                va="center", fontsize=7.5, color=s["INK"],
            )
        ax.axvline(0, color=s["INK_2"], linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(
            ["vs " + LABEL[e["comparison"].split(" - ")[1]] for e in items], fontsize=8.5
        )
        ax.set_title(s["TASK_LABEL"][task], color=s["INK"], fontsize=11)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=s["GRID"], linewidth=0.8)
        # The annotations sit to the right of each interval, so the room they
        # need is reserved rather than left to autoscaling.
        ax.set_xlim(lo_min - 0.10 * width, hi_max + 0.62 * width)
        ax.set_ylim(-0.7, len(items) - 0.3)
    np.atleast_1d(axes).ravel()[0].set_xlabel(
        "paired difference in sealed-test loss\n"
        "(Backprop-NEAT − control; negative favours Backprop-NEAT)",
        color=s["INK_2"], fontsize=8.5,
    )
    fig.suptitle(
        f"Track {track_name}: paired within-replicate sealed-test loss, "
        "95% bootstrap interval, wins out of pairs",
        color=s["INK"], fontsize=11,
    )
    return s["_save"](fig, Path(out) / f"paired-test-loss-track-{track_name.lower()}.png")


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
