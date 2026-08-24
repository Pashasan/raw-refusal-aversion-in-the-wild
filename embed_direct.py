"""Resume an mxbai embedding job using direct HTTP calls (no ollama-python),
with explicit per-request timeouts. Reads/writes the same .embed_progress.npz
format as llm_dynamics.data.clustering and finalises into the same outputs.

Usage:
  python embed_direct.py \
    --input gpt41mini_risky_add_message_df.pkl \
    --output gpt41mini_risky_add_message_df_with_users.pkl \
    --embeddings-out gpt41mini_risky_add_first_user_embeddings.npz \
    --workers 4 --timeout 30 --batch-size 200
"""
from __future__ import annotations

import argparse
import concurrent.futures
import pickle
import time
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OLLAMA_URL = "http://localhost:11434/api/embeddings"
OLLAMA_BATCH_URL = "http://localhost:11434/api/embed"  # newer endpoint accepting array input


def _first_user_rows(msg_df: pd.DataFrame) -> pd.DataFrame:
    mask = (msg_df["role"] == "user") & (msg_df["message_number"] == 1)
    return (
        msg_df.loc[mask, ["conversation_id", "content"]]
        .drop_duplicates("conversation_id")
        .reset_index(drop=True)
    )


def _load_progress(path: Path, ids: list[str]) -> tuple[list, np.ndarray]:
    """Returns (embeddings list, done_mask). Resumes if path exists and ids match."""
    if not path.exists():
        return [None] * len(ids), np.zeros(len(ids), dtype=bool)
    z = np.load(path, allow_pickle=True)
    saved_ids = list(z["conversation_ids"])
    if saved_ids != ids:
        print(f"[embed_direct] progress ids mismatch — starting fresh "
              f"({len(saved_ids):,} vs {len(ids):,})")
        return [None] * len(ids), np.zeros(len(ids), dtype=bool)
    mask = z["done_mask"]
    mat = z["embeddings"]
    embs = [mat[i].copy() if ok else None for i, ok in enumerate(mask)]
    return embs, mask


def _save_progress(path: Path, ids: list[str], embs: list, dim: int) -> None:
    mat = np.zeros((len(embs), dim), dtype=np.float32)
    mask = np.zeros(len(embs), dtype=bool)
    for i, e in enumerate(embs):
        if e is not None:
            mat[i] = e
            mask[i] = True
    np.savez_compressed(path,
        conversation_ids=np.asarray(ids, dtype=object),
        embeddings=mat,
        done_mask=mask)


MAX_PROMPT_CHARS = 500  # mxbai-embed-large 512-token context; batch /api/embed
                        # rejects above this on heavy-Unicode WildChat prompts


def _embed_one(model: str, prompt: str, timeout: float) -> np.ndarray | None:
    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = prompt[:MAX_PROMPT_CHARS]
    try:
        r = requests.post(OLLAMA_URL,
                          json={"model": model, "prompt": prompt},
                          timeout=timeout)
        r.raise_for_status()
        return np.asarray(r.json()["embedding"], dtype=np.float32)
    except Exception:  # noqa: BLE001
        return None


def _embed_batch(model: str, prompts: list[str], timeout: float) -> list[np.ndarray | None]:
    """Embed many prompts in a single Ollama /api/embed call. Returns one
    np.ndarray per input (or None for the whole batch on failure)."""
    inputs = [(p[:MAX_PROMPT_CHARS] if len(p) > MAX_PROMPT_CHARS else p) or " "
              for p in prompts]
    try:
        r = requests.post(OLLAMA_BATCH_URL,
                          json={"model": model, "input": inputs},
                          timeout=timeout)
        r.raise_for_status()
        embs = r.json().get("embeddings") or []
        if len(embs) != len(prompts):
            return [None] * len(prompts)
        return [np.asarray(e, dtype=np.float32) for e in embs]
    except Exception:  # noqa: BLE001
        return [None] * len(prompts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--embeddings-out", required=True)
    p.add_argument("--model", default="mxbai-embed-large")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--threshold", type=float, default=0.95)
    args = p.parse_args()

    msg = pickle.load(open(ROOT / args.input, "rb"))
    first = _first_user_rows(msg)
    ids = first["conversation_id"].astype(str).tolist()
    contents = first["content"].fillna("").astype(str).tolist()
    print(f"[embed_direct] {len(ids):,} first-user msgs to embed (model={args.model})")

    progress_path = ROOT / (Path(args.output).stem + ".embed_progress.npz")
    embs, done_mask = _load_progress(progress_path, ids)
    todo = [i for i in range(len(ids)) if embs[i] is None]
    print(f"[embed_direct] resumed: {int(sum(1 for e in embs if e is not None)):,} done, "
          f"{len(todo):,} remaining")

    if not todo:
        print("[embed_direct] nothing to do; finalising")
    else:
        start = time.time()
        last_save_count = sum(1 for e in embs if e is not None)
        DIM = 1024  # mxbai-embed-large
        # Use Ollama's /api/embed batched endpoint with a worker pool. Each
        # worker sends `batch_size` inputs in one HTTP call; multiple workers
        # in flight keep the GPU pipeline full.
        all_batches = []
        for b_start in range(0, len(todo), args.batch_size):
            b_end = min(b_start + args.batch_size, len(todo))
            all_batches.append(todo[b_start:b_end])

        save_every = max(1, args.workers * 2)  # save every ~2 worker-cycles
        pending = list(all_batches)
        save_counter = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            # Submit an initial wave equal to worker count
            in_flight = {}
            def submit_one(idx_pos: int):
                if idx_pos >= len(pending):
                    return
                batch = pending[idx_pos]
                prompts = [contents[i] for i in batch]
                fut = ex.submit(_embed_batch, args.model, prompts, args.timeout * 4)
                in_flight[fut] = (idx_pos, batch)
            cursor = 0
            for _ in range(args.workers):
                if cursor < len(pending):
                    submit_one(cursor); cursor += 1
            while in_flight:
                done_set, _ = concurrent.futures.wait(
                    in_flight.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in done_set:
                    idx_pos, batch = in_flight.pop(fut)
                    try:
                        results = fut.result()
                    except Exception:
                        results = [None] * len(batch)
                    for i, e in zip(batch, results):
                        embs[i] = e
                    save_counter += 1
                    if save_counter % save_every == 0:
                        _save_progress(progress_path, ids, embs, DIM)
                        done_total = sum(1 for e in embs if e is not None)
                        elapsed = time.time() - start
                        new_done = done_total - last_save_count
                        rate = new_done / elapsed if elapsed > 0 else 0.0
                        remaining = len(ids) - done_total
                        eta_min = (remaining / rate / 60) if rate > 0 else float("inf")
                        print(f"[embed_direct]   {done_total:,}/{len(ids):,}  "
                              f"rate={rate:.1f}/s  ETA {eta_min:.1f} min", flush=True)
                    if cursor < len(pending):
                        submit_one(cursor); cursor += 1
        # Final save after pool drained
        _save_progress(progress_path, ids, embs, DIM)

    # Finalise: drop None / bad rows, L2-normalise, save final outputs
    rows, kept_ids = [], []
    for cid, e in zip(ids, embs):
        if e is None or e.shape[0] < 2:
            continue
        rows.append(e)
        kept_ids.append(cid)
    if not rows:
        raise RuntimeError("No successful embeddings.")
    mat = np.stack(rows, axis=0).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.clip(norms, 1e-8, None)
    print(f"[embed_direct] kept {len(kept_ids):,} of {len(ids):,} embeddings (after dropping None)")

    # Save embeddings .npz in the same format as clustering.py outputs
    np.savez_compressed(ROOT / args.embeddings_out,
        conversation_ids=np.asarray(kept_ids, dtype=object),
        embeddings=mat)
    print(f"[embed_direct] wrote {args.embeddings_out}")

    # Build user clusters via cosine threshold + connected components
    sim = mat @ mat.T
    adj = sim >= args.threshold
    np.fill_diagonal(adj, False)
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    sp = csr_matrix(adj)
    n_comp, labels = connected_components(sp, directed=False)
    cid_to_user = {cid: int(lab) for cid, lab in zip(kept_ids, labels)}
    print(f"[embed_direct] {n_comp:,} user clusters at threshold {args.threshold}")

    # Attach user_id to msg_df, drop convs with no embedding
    msg = msg[msg["conversation_id"].astype(str).isin(set(kept_ids))].copy()
    msg["user_id"] = msg["conversation_id"].astype(str).map(cid_to_user)
    msg.to_pickle(ROOT / args.output)
    print(f"[embed_direct] wrote {args.output} ({len(msg):,} rows)")


if __name__ == "__main__":
    main()
