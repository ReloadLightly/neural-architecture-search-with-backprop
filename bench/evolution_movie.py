"""An animation of one search, so the process is visible and not only scored.

The sister repository shows GIFs of the matches its agents play. The analogue
here is the thing this search produces: a decision boundary and the graph that
computes it, both changing generation by generation.

**This is a demonstration, not evidence.** It is one run on a burned pilot seed
(9003 / 19003), outside every release, and no number from it may be cited. It
exists so a reader can see what "evolution found a topology" looks like, which
no table in this repository conveys.

The loop below is v5's ``search`` with a snapshot taken each generation: it
calls the same ``evaluate``, ``kmedoids`` and ``_reproduce``, in the same order,
with the same config, because an animation of a different algorithm would be
worse than none.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from bpneat.champions import Champion, boundary_field, topology_layout  # noqa: E402
from bpneat.evolve import Individual, InnovationRegistry, evaluate, kmedoids  # noqa: E402
from bpneat.genome import logistic_genome  # noqa: E402
from bpneat.v3.datasets import make_bundle  # noqa: E402
from bpneat.v5.conditions import config_for  # noqa: E402
from bpneat.v5.search import _reproduce  # noqa: E402

OUT = ROOT / "docs" / "figures"
from bpneat.style import (  # noqa: E402
    CLASS_COLOURS,
    DPI,
    GRID,
    INK,
    INK2,
    OP_COLOUR,
    SERIES,
    SURFACE,
    boundary_cmap,
    panel,
    style_axes,
)
BOUNDARY_CMAP = boundary_cmap()

DATASET_SEED = 9003
SEARCH_SEED = 19003
TASK = "spiral"
#: The arm that actually complexifies — an animation of the reference arm would
#: be four frames of nothing happening, which is itself a finding but a still
#: one, and §2 of the README already states it.
CONDITION = "neat_complexify_no_penalty"


def run_and_snapshot():
    """v5's search loop, with the champion captured at every generation."""
    bundle = make_bundle(TASK, seed=DATASET_SEED)
    cfg = config_for(CONDITION, TASK)
    rng = np.random.default_rng(SEARCH_SEED)
    reg = InnovationRegistry()

    pop = []
    for _ in range(cfg.population):
        g = logistic_genome(rng)
        pop.append(Individual(genome=g, weights=np.array(g.weight, dtype=np.float64)))

    champion = None
    frames = []
    for gen in range(cfg.generations + 1):
        for ind in pop:
            if not ind.evaluated:
                evaluate(ind, bundle, cfg, rng)
        best = max(pop, key=lambda i: i.fitness)
        if champion is None or best.fitness > champion.fitness:
            champion = best.copy()
        frames.append(
            {
                "generation": gen,
                "genome": champion.genome.copy(),
                "weights": champion.weights.copy(),
                "fitness": float(champion.fitness),
                "mean_nodes": float(np.mean([i.genome.n_nodes for i in pop])),
            }
        )
        print(f"  gen {gen:3d}  champion fitness {champion.fitness:+.4f}  "
              f"nodes {champion.genome.n_nodes}", flush=True)
        if gen == cfg.generations:
            break
        kmedoids(pop, cfg.n_species, rng)
        pop, *_ = _reproduce(pop, cfg, rng, reg)
    return bundle, frames


def _champ(frame, bundle) -> Champion:
    return Champion(
        task=TASK, run_id="demo", replicate=0, track="", condition=CONDITION,
        genome=frame["genome"], weights=frame["weights"], settle=True,
        dataset_seed=DATASET_SEED, test_accuracy=None, n_candidates=0, rank=0,
    )


def render(bundle, frames, out: Path, fps: int = 3) -> Path:
    fig, (ax_b, ax_n) = plt.subplots(
        1, 2, figsize=(9.2, 4.6), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.0, 1.25]},
    )
    OUT.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(out), dpi=110):
        for frame in frames:
            champ = _champ(frame, bundle)
            for ax in (ax_b, ax_n):
                ax.clear()

            field = boundary_field(champ, bundle, resolution=150)
            ax_b.imshow(field["prob"], extent=field["extent"], origin="lower",
                        cmap=BOUNDARY_CMAP, vmin=0, vmax=1,
                        interpolation="bilinear", aspect="equal")
            ax_b.contour(field["xx"], field["yy"], field["prob"], levels=[0.5],
                         colors=[INK], linewidths=1.1, alpha=0.75)
            X, y = field["train_X"], field["train_y"]
            for cls, colour in zip((0.0, 1.0), CLASS_COLOURS):
                m = y == cls
                ax_b.scatter(X[m, 0], X[m, 1], s=6, c=colour, linewidths=0.4,
                             edgecolors="white", alpha=0.9, zorder=3)
            ax_b.set_xticks([])
            ax_b.set_yticks([])
            for s in ax_b.spines.values():
                s.set_color(GRID)
            ax_b.set_title("what the champion computes", fontsize=10, color=INK)

            lay = topology_layout(champ, bundle)
            wmax = max((abs(e["weight"]) for e in lay["edges"]), default=1.0) or 1.0
            for e in lay["edges"]:
                causal = e["causal"]
                ax_n.plot(
                    [e["x0"], e["x1"]], [e["y0"], e["y1"]],
                    color=(SERIES[0] if e["weight"] >= 0 else SERIES[1])
                    if causal else GRID,
                    linewidth=(0.5 + 2.4 * abs(e["weight"]) / wmax) if causal else 0.6,
                    alpha=0.85 if causal else 0.4, zorder=1, solid_capstyle="round",
                )
            for n in lay["nodes"]:
                ax_n.scatter(
                    n["x"], n["y"], s=130 if n["structural"] else 95,
                    c=OP_COLOUR.get(n["operator"], "#9a9a95") if n["causal"] else SURFACE,
                    edgecolors=INK2 if n["causal"] else GRID, linewidths=1.0,
                    zorder=3, marker="s" if n["structural"] else "o",
                )
            ax_n.set_xlim(-0.6, max(lay["n_columns"] - 0.4, 2.0))
            ax_n.set_ylim(-1.25, 1.25)
            ax_n.axis("off")
            ax_n.set_title(
                f"the graph that computes it — "
                f"{lay['causal_hidden']} causal of {lay['represented_hidden']} hidden",
                fontsize=10, color=INK,
            )

            fig.suptitle(
                f"generation {frame['generation']:>3d} of {frames[-1]['generation']}"
                f"     ·     population mean size {frame['mean_nodes']:.1f} nodes",
                fontsize=11, color=INK,
            )
            fig.text(0.5, 0.015,
                     "demonstration run on a burned pilot seed — outside every "
                     "release, not evidence, no number from it may be cited",
                     ha="center", fontsize=8, color=INK2, style="italic")
            fig.subplots_adjust(top=0.84, bottom=0.08, left=0.03, right=0.98,
                                wspace=0.05)
            writer.grab_frame(facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)} ({len(frames)} frames)")
    return out


def main() -> int:
    print(f"demonstration run: {CONDITION} on {TASK}, seeds {DATASET_SEED}/{SEARCH_SEED}")
    bundle, frames = run_and_snapshot()
    render(bundle, frames, OUT / "evolution.gif")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
