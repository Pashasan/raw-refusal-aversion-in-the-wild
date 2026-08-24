"""Concatenate base WildChat samples with their risky-enriched additions.

For each WildChat tag, merges:
  {tag}_message_df_with_users.pkl           + {tag}_risky_add_message_df_with_users.pkl
  {tag}_refusal_data_dynamic_with_labels_gpt54.csv
    + {tag}_risky_add_refusal_data_dynamic_with_labels_gpt54.csv
  {tag}_first_user_embeddings.npz           + {tag}_risky_add_first_user_embeddings.npz

Outputs:
  {tag}_aug_message_df_with_users.pkl
  {tag}_aug_refusal_data_dynamic_with_labels_gpt54.csv
  {tag}_aug_first_user_embeddings.npz

Note the merged sample is non-random (risky-enriched). Unconditional averages
on the merged sample are biased; ATT analyses that condition on risky remain
unbiased.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

TAGS = ["gpt4o", "gpt4omini", "gpt41mini", "gpt35wc", "o1mini"]


def merge_tag(tag: str) -> None:
    print(f"\n=== {tag} ===")
    base_pkl = ROOT / f"{tag}_message_df_with_users.pkl"
    base_lab = ROOT / f"{tag}_refusal_data_dynamic_with_labels_gpt54.csv"
    base_emb = ROOT / f"{tag}_first_user_embeddings.npz"
    add_pkl  = ROOT / f"{tag}_risky_add_message_df_with_users.pkl"
    add_lab  = ROOT / f"{tag}_risky_add_refusal_data_dynamic_with_labels_gpt54.csv"
    add_emb  = ROOT / f"{tag}_risky_add_first_user_embeddings.npz"

    for p in (base_pkl, base_lab, base_emb, add_pkl, add_lab, add_emb):
        if not p.exists():
            print(f"  missing: {p.name}  — skipping {tag}")
            return

    # --- pkl merge ---
    base_msg = pickle.load(open(base_pkl, "rb"))
    add_msg  = pickle.load(open(add_pkl, "rb"))
    # Align columns (outer union, fill NaN)
    merged = pd.concat([base_msg, add_msg], ignore_index=True, sort=False)
    # Dedup on (conversation_id, message_number) in case of overlap
    before = len(merged)
    merged = merged.drop_duplicates(
        subset=["conversation_id", "message_number"], keep="first"
    ).reset_index(drop=True)
    n_base_conv = base_msg["conversation_id"].nunique()
    n_add_conv = add_msg["conversation_id"].nunique()
    n_merged_conv = merged["conversation_id"].nunique()
    print(f"  pkl: base={len(base_msg):,} rows ({n_base_conv:,} convs) + "
          f"add={len(add_msg):,} ({n_add_conv:,}) -> {len(merged):,} "
          f"({n_merged_conv:,} convs)  dropped_dup_rows={before - len(merged):,}")
    out_pkl = ROOT / f"{tag}_aug_message_df_with_users.pkl"
    merged.to_pickle(out_pkl)
    print(f"  wrote {out_pkl.name} ({out_pkl.stat().st_size/1e6:.1f} MB)")

    # --- labels merge ---
    base_df = pd.read_csv(base_lab)
    add_df  = pd.read_csv(add_lab)
    # If base labels CSV already contains risky-add rows (possible from prior collect),
    # the outer concat + dedup will just keep base's version.
    comb = pd.concat([base_df, add_df], ignore_index=True, sort=False)
    before = len(comb)
    comb = comb.drop_duplicates(
        subset=["conversation_id", "message_number"], keep="first"
    ).reset_index(drop=True)
    n_lab = comb["is_refusal_gpt54"].notna().sum()
    ref_rate = comb["is_refusal_gpt54"].mean()
    print(f"  labels: base={len(base_df):,} + add={len(add_df):,} -> {len(comb):,} rows "
          f"({n_lab:,} labelled, refusal rate {ref_rate:.4f})")
    out_lab = ROOT / f"{tag}_aug_refusal_data_dynamic_with_labels_gpt54.csv"
    comb.to_csv(out_lab, index=False)
    print(f"  wrote {out_lab.name}")

    # --- embeddings merge ---
    z1 = np.load(base_emb, allow_pickle=True)
    z2 = np.load(add_emb, allow_pickle=True)
    ids1 = list(z1["conversation_ids"])
    ids2 = list(z2["conversation_ids"])
    emb1 = z1["embeddings"]; emb2 = z2["embeddings"]
    # Concatenate; dedup by keeping first occurrence
    seen = set()
    keep_ids = []
    keep_emb_rows = []
    for src_ids, src_emb in [(ids1, emb1), (ids2, emb2)]:
        for i, cid in enumerate(src_ids):
            if cid in seen:
                continue
            seen.add(cid)
            keep_ids.append(cid)
            keep_emb_rows.append(src_emb[i])
    keep_emb = np.stack(keep_emb_rows, axis=0)
    # Keep done_mask = all True since these came from completed files
    done_mask = np.ones(len(keep_ids), dtype=bool)
    print(f"  embeds: base={len(ids1):,} + add={len(ids2):,} -> "
          f"{len(keep_ids):,} unique convs  dim={keep_emb.shape[1]}")
    out_emb = ROOT / f"{tag}_aug_first_user_embeddings.npz"
    np.savez_compressed(out_emb,
        conversation_ids=np.asarray(keep_ids, dtype=object),
        embeddings=keep_emb.astype(np.float32),
        done_mask=done_mask)
    print(f"  wrote {out_emb.name} ({out_emb.stat().st_size/1e6:.1f} MB)")


def main() -> None:
    for tag in TAGS:
        merge_tag(tag)


if __name__ == "__main__":
    main()
