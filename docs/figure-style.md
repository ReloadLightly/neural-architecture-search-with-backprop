# The figure style

Every figure in this repository is drawn in the style of the sibling project
[`competitive-coevolution-of-slimes`](https://github.com/ReloadLightly/competitive-coevolution-of-slimes).
The two are one body of work and should read as one. This page is the contract;
`src/bpneat/style.py` is its implementation and `tests/test_style.py` enforces it.

## The register

White ground. No frame, no border, no page rule. Top and right spines off, a
very light grid (`#E6E6E6`) behind the data, `axisbelow`. Stock sans — whatever
matplotlib ships — at 8.5pt, with 9.5pt titles, 7.5pt ticks and 7.5pt legends.
Legends have no box. Figures are saved at 200 dpi with a tight bounding box, so
there is no margin and nothing is drawn that is not either data or the furniture
needed to read it.

Titles are **left-aligned sentences** that say what the panel shows, not labels.
Axis labels are lowercase.

## The palette

Twelve hues, taken unchanged from the sibling project:

| | | | |
|---|---|---|---|
| `#222222` ink | `#3B6EA8` slate | `#1F7A5A` pine | `#E08B3C` amber |
| `#B0413E` brick | `#8C5A2B` umber | `#7A5EA6` violet | `#C1445E` rose |
| `#4C956C` sage | `#946A9E` mauve | `#8A6A55` taupe | `#4FA3B8` teal |

Nothing in this repository invents a hue. It only chooses which of the twelve a
series gets, and it chooses **by what the series is, never by its position in a
list**. Every comparison here is between something we searched for and something
we fixed in advance, so:

| role | hue | meaning |
|---|---|---|
| `search_primary` | slate | the algorithm under study |
| `search_secondary` | sage | a second search, or a variant of the first |
| `search_null` | mauve | the same space, sampled instead of searched |
| `control_starved` | amber | a fixed architecture denied the search's budget |
| `control_matched` | brick | a fixed architecture given that budget |
| `control_best` | rose | the strongest fixed control |
| `reference` | ink | a published target, or a verdict below significance |

Which six of the twelve was decided by searching all C(11,6) subsets for the one
with the largest worst-pair separation under normal, protanopic and deuteranopic
vision. The measured floors — 15.7 / 14.0 / 11.4 ΔE — are pinned as a ratchet in
`tests/test_style.py`. The nine operator hues were chosen the same way.

Colour never carries identity alone: every figure either directly labels its
marks or ships a CSV table view beside it in the release.

## The idioms

- a dashed grey parity line to measure everything against — `style.parity(ax)`
- one dot per replicate with the group median as a short bar — `style.strip(...)`
- marks carry a white keyline, so overlaps stay readable — `style.dot(...)`
- a median line over an inter-quartile band, rather than a mean with error caps
- direct annotation in the colour of the thing it points at

## Size

**No figure is wider than 6.8 inches.** GitHub renders a README image at about
870 px; a 13-inch figure at 200 dpi is 2720 px, so it displays at a third of its
size and its 7.5pt tick labels arrive at about 2pt. That is the single largest
cause of an unreadable figure in this project's history, and it is not fixable
by choosing better colours.

Panels therefore **stack downwards, never sideways**. A figure with five task
panels is five rows, not five columns. Where a figure needs a summary and its
parts, it takes the sibling project's shape: one wide panel on top, a row of
small strip panels beneath.

## What is drawn at all

**Every committed figure is a measured result from a committed release record.**
Not the input datasets, not an illustrative run, not a pilot seed. Three figures
were removed under this rule: the five task geometries (the problem setup), an
animation of one search on a burned pilot seed (explicitly not evidence), and
two specimen plates of champion genomes (a portrait, not a finding).
`tests/test_readme_figures.py` keeps them gone, and fails on any figure
committed to `docs/figures/` that the README does not show.

## What is deliberately not restyled

`results/backprop-neat-v1/figures/` belongs to an invalidated release and
`results/backprop-neat-v2/` carries errata. Both are frozen evidence, and a
frozen release should look like what was released. Nothing in the README or the
paper shows them.
