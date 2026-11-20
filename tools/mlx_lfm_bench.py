"""LFM2.5 on the Mac's GPU (engine/models/lfm2_moe/mlx_board.py): greedy text, speed, and the logits against
mlx-lm's own lfm2_moe.

    .venv-mac/bin/python tools/mlx_lfm_bench.py <checkpoint dir> [--tokens 256] [--prompt-len 512] [--check]

Prints the greedy continuation of a chat prompt, the prefill rate of a `--prompt-len` prompt and the decode
rate of `--tokens` greedy tokens (one request, plain decode, no drafting). `--check` also loads mlx-lm's
model on the same checkpoint and compares the last-row logits of a prompt (max |diff|, top-1, top-5 overlap).
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from engine.models.lfm2_moe import mlx_board  # noqa: E402

CHAT = [{"role": "user", "content": "Explain in three short paragraphs why the sky is blue."}]


def greedy(board, cfg, ids: list[int], n: int, cap: int):
    """(tokens, prefill seconds, decode seconds) of a greedy run on the pipelined loop."""
    st = board.new_state(cap)
    t0 = time.perf_counter()
    lg = board.prefill(ids, st, last_only=True)
    first = int(mx.argmax(lg[-1]).item())
    t1 = time.perf_counter()
    out = [first]
    if first not in cfg.eos_token_ids:
        out += list(board.generate(first, st, n - 1, stop=set(cfg.eos_token_ids)))
    return out, t1 - t0, time.perf_counter() - t1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.path)
    t0 = time.perf_counter()
    cfg, board = mlx_board.load(a.path)
    print(f"load {time.perf_counter() - t0:.1f} s, peak {mx.get_peak_memory() / 1e9:.1f} GB")

    text = tok.apply_chat_template(CHAT, add_generation_prompt=True, tokenize=False)
    ids = tok.encode(text, add_special_tokens=False)
    greedy(board, cfg, ids, 4, 4096)                         # warm-up: kernels compiled
    out, _, dt = greedy(board, cfg, ids, a.tokens, 4096)
    print("---\n" + tok.decode(out) + "\n---")
    print(f"decode: {len(out) - 1} tokens in {dt:.2f} s = {(len(out) - 1) / dt:.1f} tok/s")

    long_ids = (tok.encode(open(__file__).read()) * 64)[:a.prompt_len]
    _, pf, _ = greedy(board, cfg, long_ids, 1, a.prompt_len + 64)
    print(f"prefill: {a.prompt_len} tokens in {pf:.2f} s = {a.prompt_len / pf:.0f} tok/s")
    print(f"peak memory {mx.get_peak_memory() / 1e9:.1f} GB")

    if a.check:
        st = board.new_state(len(ids) + 8)
        mine = np.array(board.prefill(ids, st, last_only=True)[-1])
        del board, st
        gc.collect()
        mx.clear_cache()
        from mlx_lm import load as lm_load
        model, _ = lm_load(a.path)
        theirs = np.array(model(mx.array([ids]))[0, -1].astype(mx.float32))
        t5m, t5t = set(np.argsort(-mine)[:5]), set(np.argsort(-theirs)[:5])
        print(f"vs mlx-lm: max |diff| {np.abs(mine - theirs).max():.3f} (logits up to {np.abs(theirs).max():.1f}), "
              f"top-1 {'same' if mine.argmax() == theirs.argmax() else 'DIFFERENT'}, top-5 overlap {len(t5m & t5t)}/5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
