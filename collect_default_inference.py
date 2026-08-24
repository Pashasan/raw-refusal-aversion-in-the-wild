"""Assemble the default-inference JSON the figure scripts read.

The default inference regime clusters standard errors on hashed-IP users
for WildChat cells (--cluster ip) and on near-duplicate prompt groups for
LMSYS cells (--cluster promptcluster); see the paper's Appendix on
inference under dependence. Run aipw_with_gpt54.py with each cluster flag
first (per-cell JSONs land in <outdir>/cells/), then this script to merge
them into output/comparison/aipw_clustered/default_inference.json.

Usage:
  python collect_default_inference.py --cells-dir output/comparison/aipw_clustered/cells
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Preferred cluster mode per cell: user (hashed-IP) clusters wherever the
# source data identify users; prompt-group clusters otherwise.
PREFERENCE = ("cluster-ip", "cluster-promptcluster")
MODE_NAME = {"cluster-ip": "ip", "cluster-promptcluster": "prompt"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells-dir", default="output/comparison/aipw_clustered/cells",
                    help="directory of per-cell JSONs written by "
                         "aipw_with_gpt54.py --cluster ... --outdir ...")
    ap.add_argument("--out", default="output/comparison/aipw_clustered/default_inference.json")
    args = ap.parse_args()

    cells_dir = ROOT / args.cells_dir
    records: dict[str, dict] = {}
    for variant in PREFERENCE:
        for fp in sorted(cells_dir.glob(f"{variant}__*.json")):
            tag = fp.stem.split("__", 1)[1]
            if tag in records:
                continue  # a preferred-mode record already exists
            c = json.loads(fp.read_text())
            records[tag] = {
                "cluster_mode": MODE_NAME[variant],
                "att_pp": c["att_pp"],
                "def_lo": c["att_ci95_cluster_lo"],
                "def_hi": c["att_ci95_cluster_hi"],
                "n_clusters": c.get("n_clusters"),
            }

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=1))
    print(f"wrote {out} ({len(records)} cells)")


if __name__ == "__main__":
    main()
