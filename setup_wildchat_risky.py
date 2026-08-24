"""Risky-enriched WildChat additions: for a given model, extract *new*
conversations (not already in the base sample) whose first-user prompt is
risky (max_concern_score > 0.01). Produces:

  {tag}_risky_add_message_df.pkl
  {tag}_risky_add_refusal_data_dynamic.csv

After labelling we concatenate with the base sample to get an expanded
risky-treated pool. The sampling is *non-uniform* (risky-enriched), so any
unconditional mean across the concat would be biased; ATT analyses that
condition on risky remain unbiased.

Usage:
  python setup_wildchat_risky.py --tag gpt41mini --model gpt-4.1-mini-2025-04-14 --target 30000
"""
from __future__ import annotations

import argparse
import pickle
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


def _first_user_max_score(row: pd.Series) -> float:
    convo = row["conversation"]; mods = row["openai_moderation"]
    if hasattr(convo, "tolist"): convo = convo.tolist()
    if hasattr(mods, "tolist"): mods = mods.tolist()
    for m, md in zip(convo, mods):
        role = m.get("role", "") if isinstance(m, dict) else ""
        if role == "user":
            scores = md.get("category_scores", {}) or {}
            return float(max(scores.values())) if scores else 0.0
    return 0.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--language", default="English")
    p.add_argument("--base-pkl", default=None,
                   help="Existing {tag}_message_df.pkl to deduplicate against. "
                        "Omit for new models with no base sample (extracts all "
                        "risky convs).")
    p.add_argument("--target", type=int, default=30000,
                   help="Max new risky convs to add")
    p.add_argument("--risk-thresh", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=43)
    args = p.parse_args()

    if args.base_pkl is None:
        used_hashes: set = set()
        print(f"[{args.tag}] no base pkl — extracting all risky convs")
    else:
        base_pkl = ROOT / args.base_pkl
        if not base_pkl.exists():
            sys.exit(f"base pkl not found: {base_pkl}")
        base_msg = pickle.load(open(base_pkl, "rb"))
        used_hashes = set(base_msg["conversation_id"].unique())
        print(f"[{args.tag}] base sample has {len(used_hashes):,} conv_hashes")

    shards = sorted(WILDCHAT.glob("train-*.parquet"))
    print(f"[{args.tag}] scanning {len(shards)} shards for model={args.model}")

    collected: list[pd.DataFrame] = []
    n_risky_seen = 0
    for i, shard in enumerate(shards):
        df = pd.read_parquet(
            shard,
            columns=["conversation_hash", "model", "language", "conversation",
                     "turn", "openai_moderation"],
        )
        df = df[(df["model"] == args.model) & (df["language"] == args.language)]
        if df.empty:
            if (i+1) % 30 == 0 or i == len(shards)-1:
                print(f"  shard {i+1}/{len(shards)} (running new risky: 0)")
            continue
        df = df[~df["conversation_hash"].isin(used_hashes)]
        if df.empty:
            if (i+1) % 30 == 0 or i == len(shards)-1:
                print(f"  shard {i+1}/{len(shards)} (running new risky: {n_risky_seen})")
            continue
        # Filter to risky first-user-prompt convs
        df = df.assign(_fu_score=df.apply(_first_user_max_score, axis=1))
        df = df[df["_fu_score"] > args.risk_thresh].drop(columns=["_fu_score"])
        if len(df):
            collected.append(df)
            n_risky_seen += len(df)
        if (i+1) % 20 == 0 or i == len(shards)-1:
            print(f"  shard {i+1}/{len(shards)} (running new risky: {n_risky_seen:,})")

    if not collected:
        print(f"[{args.tag}] no new risky convs found")
        return 1
    raw = pd.concat(collected, ignore_index=True).drop_duplicates(subset="conversation_hash")
    print(f"[{args.tag}] total new risky unique convs available: {len(raw):,}")
    if len(raw) > args.target:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(raw), size=args.target, replace=False)
        raw = raw.iloc[sorted(keep)].reset_index(drop=True)
        print(f"[{args.tag}] capped at {args.target:,} (seed={args.seed})")

    rows: list[dict] = []
    for _, r in raw.iterrows():
        rows.extend(_explode_one(r))
    msg = pd.DataFrame(rows)
    score_cols = [c for c in msg.columns if c.endswith("_score")]
    msg["max_concern_score"] = msg[score_cols].max(axis=1)
    print(f"[{args.tag}] new msg rows: {len(msg):,}  "
          f"({(msg['role']=='user').sum():,} user / "
          f"{(msg['role']=='assistant').sum():,} assistant)")

    pkl_out = ROOT / f"{args.tag}_risky_add_message_df.pkl"
    msg.to_pickle(pkl_out)
    print(f"[{args.tag}] wrote {pkl_out.name}")

    ast = msg.loc[msg["role"] == "assistant",
                  ["conversation_id", "content", "role", "message_number"]]
    csv_out = ROOT / f"{args.tag}_risky_add_refusal_data_dynamic.csv"
    ast.to_csv(csv_out, index=False)
    print(f"[{args.tag}] wrote {csv_out.name} ({len(ast):,} assistant rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
