"""Generic WildChat setup: build {tag}_message_df.pkl + {tag}_refusal_data_dynamic.csv
for any WildChat model.

Usage:
  python setup_wildchat.py --tag gpt4omini --model gpt-4o-mini-2024-07-18
  python setup_wildchat.py --tag gpt41mini --model gpt-4.1-mini-2025-04-14 --sample 22000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WILDCHAT = Path("data/wildchat")  # local WildChat-4.8M-Full parquet directory


def _explode_one(row: pd.Series) -> list[dict]:
    convo = row["conversation"]
    mods = row["openai_moderation"]
    if hasattr(convo, "tolist"):
        convo = convo.tolist()
    if hasattr(mods, "tolist"):
        mods = mods.tolist()
    if len(convo) != len(mods):
        return []
    total = len(convo)
    out = []
    for idx, (msg, mod) in enumerate(zip(convo, mods), start=1):
        content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        role = msg.get("role", "unknown") if isinstance(msg, dict) else "unknown"
        scores = mod.get("category_scores", {}) or {}
        out.append({
            "conversation_id": row["conversation_hash"],
            "model": row["model"],
            "language": row["language"],
            "role": role,
            "content": content,
            "message_number": idx,
            "total_messages": total,
            "flagged": mod.get("flagged", False),
            **{f"{k.replace('/', '_')}_score": float(v) for k, v in scores.items()},
        })
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--language", default="English")
    p.add_argument("--sample", type=int, default=22000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    shards = sorted(WILDCHAT.glob("train-*.parquet"))
    print(f"[wc:{args.tag}] scanning {len(shards)} shards for model={args.model}")

    parts: list[pd.DataFrame] = []
    n_so_far = 0
    for i, shard in enumerate(shards):
        sm = pd.read_parquet(
            shard,
            columns=["conversation_hash", "model", "language", "conversation",
                     "turn", "openai_moderation"],
        )
        sm = sm[(sm["model"] == args.model) & (sm["language"] == args.language)]
        if len(sm):
            parts.append(sm)
            n_so_far += len(sm)
        if (i + 1) % 20 == 0 or i == len(shards) - 1:
            print(f"  shard {i+1}/{len(shards)} — running total: {n_so_far:,}")

    if not parts:
        print(f"[wc:{args.tag}] no matches — aborting")
        return 1

    raw = pd.concat(parts, ignore_index=True)
    # De-dupe on conversation_hash (WildChat can repeat the same conversation
    # across shards — we saw this with gpt4o).
    before = len(raw)
    raw = raw.drop_duplicates(subset="conversation_hash").reset_index(drop=True)
    if len(raw) < before:
        print(f"[wc:{args.tag}] dropped {before - len(raw):,} duplicate conversation_hash rows")
    print(f"[wc:{args.tag}] total unique convos: {len(raw):,}")

    rng = np.random.default_rng(args.seed)
    if len(raw) > args.sample:
        keep = rng.choice(len(raw), size=args.sample, replace=False)
        raw = raw.iloc[sorted(keep)].reset_index(drop=True)
        print(f"[wc:{args.tag}] subsampled to {args.sample:,} (seed={args.seed})")
    print(f"[wc:{args.tag}] turn: mean={raw['turn'].mean():.2f} "
          f"multi-turn(>=2)={(raw['turn']>=2).mean():.2%}")

    print(f"[wc:{args.tag}] exploding into per-message rows…")
    rows: list[dict] = []
    for _, r in raw.iterrows():
        rows.extend(_explode_one(r))
    msg = pd.DataFrame(rows)
    score_cols = [c for c in msg.columns if c.endswith("_score")]
    msg["max_concern_score"] = msg[score_cols].max(axis=1)
    print(f"[wc:{args.tag}] msg_df: {len(msg):,} rows "
          f"({(msg['role']=='user').sum():,} user / "
          f"{(msg['role']=='assistant').sum():,} assistant)")

    pkl = ROOT / f"{args.tag}_message_df.pkl"
    msg.to_pickle(pkl)
    print(f"[wc:{args.tag}] wrote {pkl.name} ({pkl.stat().st_size/1e6:.1f} MB)")

    ast = msg.loc[msg["role"] == "assistant",
                  ["conversation_id", "content", "role", "message_number"]]
    out_csv = ROOT / f"{args.tag}_refusal_data_dynamic.csv"
    ast.to_csv(out_csv, index=False)
    print(f"[wc:{args.tag}] wrote {out_csv.name} ({len(ast):,} assistant rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
