"""Where each looming cell looks: receptive fields of the 210 LPLC2 and 104 LC4 cells from FlyWire's visual column map.

Stretch C, stage 1. Until now every alarm cell got the same input rate. To let different layouts (a spike ahead,
a tower above, a pit below) drive different cells, each cell needs its own patch of the visual field.

How (all from FlyWire FAFB v783; codex/ = the Codex downloads, gitignored):
  1. columns  codex/column_assignment.csv.gz puts 45,528 columnar neurons (T4, T5, Tm, Mi1, ...) on each eye's
              hexagonal grid. Flat map: X = sqrt(3) * x, Y = y / 2, in column spacings (~4.8 degrees of view).
  2. inputs   a cell looks where its columnar inputs look: its receptive-field centre is the synapse-weighted mean of
              its presynaptic partners' columns (data/Connectivity_783.parquet), its size their weighted spread.
  3. axes     which way is up and front was measured, not assumed (python -m brain.receptive axes):
              +Y = up: medulla columns (Mi1 synapse positions, codex/synapse_coordinates.csv.gz) move ventrally in the
                 brain as Y falls (r = 1.00 both eyes), and the optic lobe keeps up as up.
              +X = front: medulla columns move posteriorly as X grows (r = 0.98), and the first optic chiasm maps the
                 eye's front onto the back of the medulla. Brain axes checked on landmarks: olfactory receptor
                 neurons (front) have small z, Kenyon cells (back, top) large z and small y.
  4. angles   azimuth 0 = the eye's front edge (the frontmost columns), growing towards the side; elevation 0 = the
              eye's mean column height. 4.8 degrees per column spacing (Drosophila interommatidial angle ~4.5-5).
              Rough: the real eye is curved and its front edges overlap a little (binocular zone); good enough to
              say which cells see straight ahead, above or below.

    python -m brain.receptive          # -> data/receptive_fields.csv and figures/receptive_fields.svg
    python -m brain.receptive axes     # re-measure the up/front axes from synapse positions (~1 min)
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from brain.paths import DATA, FIGURES, ROOT

CODEX = ROOT / "codex"
OUT = DATA / "receptive_fields.csv"
DEG_PER_COLUMN = 4.8
TYPES = ("LPLC2", "LC4")


def columns() -> pd.DataFrame:
    col = pd.read_csv(CODEX / "column_assignment.csv.gz", dtype={"root_id": str})
    col["X"], col["Y"] = np.sqrt(3) * col.x, col.y / 2
    return col


def receptive_fields() -> pd.DataFrame:
    """One row per LPLC2/LC4 cell: eye, centre (X, Y), spread, azimuth/elevation in degrees, input synapses used."""
    col = columns()
    ann = pd.read_csv(DATA / "neuron_annotations.tsv", sep="\t", usecols=["root_id", "cell_type", "side"],
                      dtype={"root_id": str})
    cells = ann[ann.cell_type.isin(TYPES)]
    con = pd.read_parquet(DATA / "Connectivity_783.parquet", columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity"])
    con = con.astype({"Presynaptic_ID": str, "Postsynaptic_ID": str})
    m = (con.merge(cells.rename(columns={"root_id": "Postsynaptic_ID"}), on="Postsynaptic_ID")
            .merge(col[["root_id", "hemisphere", "type", "X", "Y"]].rename(
                columns={"root_id": "Presynaptic_ID", "type": "pre_type"}), on="Presynaptic_ID"))
    front = col.groupby("hemisphere").X.quantile(0.99)       # the eye's front edge (robust to a stray column)
    middle = col.groupby("hemisphere").Y.mean()
    rows = []
    for rid, g in m.groupby("Postsynaptic_ID"):
        eye = g.hemisphere.value_counts().index[0]            # a cell's columns all come from one eye
        g = g[g.hemisphere == eye]
        w = g.Connectivity.to_numpy(float)
        cx, cy = np.average(g.X, weights=w), np.average(g.Y, weights=w)
        spread = np.sqrt(np.average((g.X - cx) ** 2 + (g.Y - cy) ** 2, weights=w))
        rows.append({"root_id": rid, "cell_type": g.cell_type.iloc[0], "eye": eye, "X": cx, "Y": cy,
                     "spread": spread, "azimuth": (front[eye] - cx) * DEG_PER_COLUMN,
                     "elevation": (cy - middle[eye]) * DEG_PER_COLUMN, "radius": spread * DEG_PER_COLUMN,
                     "synapses": int(w.sum()), "top_input": g.groupby("pre_type").Connectivity.sum().idxmax()})
    rf = pd.DataFrame(rows)
    missing = set(cells.root_id) - set(rf.root_id)
    if missing:
        print(f"warning: {len(missing)} cells have no columnar inputs and get no receptive field")
    return rf


def axes() -> None:
    """Re-measure how the column map's X and Y run through the brain, in the medulla (Mi1 synapses)."""
    col = columns()
    mi1 = col[col.type == "Mi1"].set_index("root_id")
    ids, parts = set(mi1.index), []
    for chunk in pd.read_csv(CODEX / "synapse_coordinates.csv.gz", chunksize=5_000_000,
                             dtype={"pre_root_id": str, "post_root_id": str}):
        chunk["pre_root_id"] = chunk["pre_root_id"].ffill()   # the file only names a neuron on its first synapse
        c = chunk[chunk.pre_root_id.isin(ids)]
        if len(c):
            parts.append(c.groupby("pre_root_id")[["x", "y", "z"]].agg(["sum", "count"]))
    g = pd.concat(parts).groupby(level=0).sum()
    pos = pd.DataFrame({k: g[(k, "sum")] / g[(k, "count")] for k in "xyz"}).join(mi1[["hemisphere", "X", "Y"]])
    for eye in ("right", "left"):
        p = pos[pos.hemisphere == eye]
        a = np.c_[p.X, p.Y, np.ones(len(p))]
        for k in "yz":
            c, *_ = np.linalg.lstsq(a, p[k], rcond=None)
            r = np.corrcoef(a[:, :2] @ c[:2], p[k])[0, 1]
            print(f"{eye} eye, medulla: brain {k} changes {c[0]:6.0f} nm per X, {c[1]:6.0f} nm per Y (r = {r:.2f})")
    print("brain y grows ventrally and z posteriorly (FAFB), so: +Y = up; +X = back of the medulla = front of view")


def chart(rf: pd.DataFrame) -> str:
    """Receptive-field centres on the two eyes (azimuth from the front edge, elevation)."""
    W, H, L, T = 760, 380, 50, 40
    sx = lambda az, eye: (L + 300 - az * 2.2) if eye == "left" else (L + 360 + az * 2.2)
    sy = lambda el: T + 160 - el * 2.2
    colour = {"LPLC2": "#d9480f", "LC4": "#1971c2"}
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#222222">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="{L}" y="22" font-size="15" font-weight="bold">Where the looming cells look (FlyWire column map)</text>',
         f'<line x1="{L + 330}" x2="{L + 330}" y1="{T + 10}" y2="{H - 50}" stroke="#adb5bd" stroke-dasharray="3 3"/>',
         f'<text x="{L + 330}" y="{H - 34}" text-anchor="middle" fill="#868e96">straight ahead</text>',
         f'<text x="{L + 150}" y="{H - 14}" text-anchor="middle">left eye</text>',
         f'<text x="{L + 510}" y="{H - 14}" text-anchor="middle">right eye</text>',
         f'<line x1="{L}" x2="{W - 20}" y1="{sy(0):.0f}" y2="{sy(0):.0f}" stroke="#e9ecef"/>',
         f'<text x="{L - 6}" y="{sy(0) + 4:.0f}" text-anchor="end" fill="#868e96">0°</text>']
    for _, r in rf.iterrows():
        s.append(f'<circle cx="{sx(r.azimuth, r.eye):.1f}" cy="{sy(r.elevation):.1f}" r="3" '
                 f'fill="{colour[r.cell_type]}" fill-opacity="0.7"/>')
    for i, (t, c) in enumerate(colour.items()):
        s.append(f'<circle cx="{W - 120}" cy="{T + 10 + 18 * i}" r="5" fill="{c}"/>'
                 f'<text x="{W - 108}" y="{T + 14 + 18 * i}">{t}</text>')
    s.append("</svg>")
    return "\n".join(s) + "\n"


def main() -> int:
    if sys.argv[1:2] == ["axes"]:
        axes()
        return 0
    rf = receptive_fields()
    rf.to_csv(OUT, index=False)
    for t in TYPES:
        g = rf[rf.cell_type == t]
        print(f"{t}: {len(g)} cells; azimuth {g.azimuth.min():.0f}-{g.azimuth.max():.0f} deg from the front edge, "
              f"elevation {g.elevation.min():.0f} to {g.elevation.max():.0f}; receptive-field radius (input spread) "
              f"median {g.radius.median():.0f} deg; inputs mostly {g.top_input.value_counts().head(3).to_dict()}")
    front = rf[rf.azimuth < 30]
    print(f"cells looking within 30 deg of straight ahead: {len(front)} "
          f"({front.cell_type.value_counts().to_dict()}), elevations {front.elevation.min():.0f} to {front.elevation.max():.0f}")
    (FIGURES / "receptive_fields.svg").write_text(chart(rf))
    print(f"wrote {OUT} and {FIGURES / 'receptive_fields.svg'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
