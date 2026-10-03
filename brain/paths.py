"""Where things live. The MLX simulator is a sibling checkout, not part of this repo."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Kisame76/drosophila-brain-mlx, cloned next to this repo (override with DROSO_MLX).
MLX_REPO = Path(os.environ.get("DROSO_MLX", ROOT.parent / "drosophila-brain-mlx"))

# Packs are compiled connectomes (~120 MB each), kept out of git.
PACKS = DATA / "pack"
PACK_V783 = PACKS / "v783"
PACK_V630 = MLX_REPO / "data" / "pack" / "v630"   # only for checking against the published v630 results


def shuffled(pack: Path, seed: int) -> Path:
    return pack.parent / f"{pack.name}-shuffled-seed{seed}"
