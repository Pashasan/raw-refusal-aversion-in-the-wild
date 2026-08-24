"""Step 1: build claude-1 message dataframe from LMSYS-Chat-1M parquets,
plus the assistant-only CSV the refusal labeller consumes.

Outputs:
  claude1_message_df.pkl
  claude1_refusal_data_dynamic.csv
"""
from __future__ import annotations

from pathlib import Path

from llm_dynamics.data.load import (
    build_messages_raw, extract_assistant_messages_for_labelling,
)

ROOT = Path(__file__).resolve().parent


def main() -> None:
    print("[setup_claude1] reading parquets, filtering model=claude-1, language=English")
    msg = build_messages_raw(model="claude-1", language="English")
    print(f"[setup_claude1] built msg_df: {len(msg):,} rows, "
          f"{msg['conversation_id'].nunique():,} conversations")
    pkl = ROOT / "claude1_message_df.pkl"
    msg.to_pickle(pkl)
    print(f"[setup_claude1] wrote {pkl.name}")

    csv = ROOT / "claude1_refusal_data_dynamic.csv"
    extract_assistant_messages_for_labelling(msg, out_csv=csv)
    print(f"[setup_claude1] wrote {csv.name} "
          f"({(msg['role']=='assistant').sum():,} assistant rows)")


if __name__ == "__main__":
    main()
