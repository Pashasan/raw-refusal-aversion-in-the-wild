"""Build a gpt-4o-2024-08-06 message dataframe from WildChat parquets.

WildChat (Aug 2025 dump) parquet shards live under WILDCHAT (set below).
Schema differs from LMSYS-Chat-1M only in the conversation key
(``conversation_hash`` instead of ``conversation_id``) and in carrying a
pre-computed ``openai_moderation`` field (same v2 schema as the API).

Outputs:
  gpt4o_message_df.pkl
  gpt4o_refusal_data_dynamic.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WILDCHAT = Path("data/wildchat")  # local WildChat-4.8M-Full parquet directory

MODEL = "gpt-4o-2024-08-06"
LANGUAGE = "English"
N_SAMPLE = 22_000
SEED = 42


def _explode_one(row: pd.Series) -> list[dict]:
    """Mirror llm_dynamics.data.load.build_messages_raw row-explosion logic,
    adapted for WildChat (conversation_hash → conversation_id)."""
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


def main() -> None:
    shards = sorted(WILDCHAT.glob("train-*.parquet"))
    print(f"[gpt4o] scanning {len(shards)} shards for model={MODEL}, language={LANGUAGE}")

    parts: list[pd.DataFrame] = []
    n_so_far = 0
    for i, shard in enumerate(shards):
        sm = pd.read_parquet(
            shard,
            columns=["conversation_hash", "model", "language", "conversation",
                     "turn", "openai_moderation"],
        )
        sm = sm[(sm["model"] == MODEL) & (sm["language"] == LANGUAGE)]
        if len(sm):
            parts.append(sm)
            n_so_far += len(sm)
        if (i + 1) % 10 == 0 or i == len(shards) - 1:
            print(f"  shard {i+1}/{len(shards)} — running total: {n_so_far:,}")

    raw = pd.concat(parts, ignore_index=True)
    print(f"[gpt4o] total convos found: {len(raw):,}")

    rng = np.random.default_rng(SEED)
    if len(raw) > N_SAMPLE:
        keep = rng.choice(len(raw), size=N_SAMPLE, replace=False)
        raw = raw.iloc[sorted(keep)].reset_index(drop=True)
        print(f"[gpt4o] subsampled to {N_SAMPLE:,} (seed={SEED})")
    print(f"[gpt4o] turn distribution after sample: "
          f"mean={raw['turn'].mean():.2f}  multi-turn(>=2)={(raw['turn']>=2).mean():.2%}")

    print("[gpt4o] exploding into per-message rows…")
    rows: list[dict] = []
    for _, r in raw.iterrows():
        rows.extend(_explode_one(r))
    msg = pd.DataFrame(rows)
    score_cols = [c for c in msg.columns if c.endswith("_score")]
    msg["max_concern_score"] = msg[score_cols].max(axis=1)
    print(f"[gpt4o] msg_df: {len(msg):,} rows  "
          f"({(msg['role']=='user').sum():,} user / "
          f"{(msg['role']=='assistant').sum():,} assistant)  "
          f"unique convs: {msg['conversation_id'].nunique():,}")

    msg.to_pickle(ROOT / "gpt4o_message_df.pkl")
    print(f"[gpt4o] wrote gpt4o_message_df.pkl ({(ROOT/'gpt4o_message_df.pkl').stat().st_size/1e6:.1f} MB)")

    ast = msg.loc[msg["role"] == "assistant",
                  ["conversation_id", "content", "role", "message_number"]]
    out_csv = ROOT / "gpt4o_refusal_data_dynamic.csv"
    ast.to_csv(out_csv, index=False)
    print(f"[gpt4o] wrote {out_csv.name} ({len(ast):,} assistant rows)")


if __name__ == "__main__":
    sys.exit(main())
