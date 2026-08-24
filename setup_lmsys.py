"""Generic LMSYS-Chat-1M setup: build {tag}_message_df.pkl + {tag}_refusal_data_dynamic.csv
for any model name.

Usage:
  python setup_lmsys.py --tag vicuna13b --model vicuna-13b
  python setup_lmsys.py --tag koala13b --model koala-13b --sample 22000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from llm_dynamics.data.load import (
    build_messages_raw, extract_assistant_messages_for_labelling,
)

ROOT = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="Short identifier used in output filenames")
    p.add_argument("--model", required=True, help="LMSYS 'model' column value")
    p.add_argument("--language", default="English")
    p.add_argument("--sample", type=int, default=22000,
                   help="Max conversations to keep (default 22000)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"[setup_lmsys:{args.tag}] model={args.model} language={args.language}")
    msg = build_messages_raw(model=args.model, language=args.language)
    n_conv = msg["conversation_id"].nunique()
    print(f"[setup_lmsys:{args.tag}] built msg_df: {len(msg):,} rows, {n_conv:,} conversations")

    if n_conv > args.sample:
        rng = np.random.default_rng(args.seed)
        keep_ids = rng.choice(msg["conversation_id"].unique(), size=args.sample, replace=False)
        msg = msg[msg["conversation_id"].isin(keep_ids)].reset_index(drop=True)
        print(f"[setup_lmsys:{args.tag}] subsampled to {args.sample:,} convs "
              f"({len(msg):,} rows, seed={args.seed})")

    pkl = ROOT / f"{args.tag}_message_df.pkl"
    msg.to_pickle(pkl)
    print(f"[setup_lmsys:{args.tag}] wrote {pkl.name} ({pkl.stat().st_size/1e6:.1f} MB)")

    csv = ROOT / f"{args.tag}_refusal_data_dynamic.csv"
    extract_assistant_messages_for_labelling(msg, out_csv=csv)
    n_asst = (msg["role"] == "assistant").sum()
    print(f"[setup_lmsys:{args.tag}] wrote {csv.name} ({n_asst:,} assistant rows)")


if __name__ == "__main__":
    main()
