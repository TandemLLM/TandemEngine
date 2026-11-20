"""StairCut's price file for LFM2.5 on Apple silicon (engine/models/lfm2_moe/mlx_board.py), in the format of
ops/prices/lfm2.5-24b-a2b/<profile>.dgx-spark.json:

    .venv-mac/bin/python tools/mlx_lfm_prices.py <checkpoint dir> [--profile lfm25-bf16] [--write]

At each context class (0: a 512-token prompt; 16384: a prompt of 16,384 tokens) it times a chain verify of
1..16 rows (`rows` + the eval, the median of `--reps`, the state put back between reps) and the plain decode
step (`decode`, ms a token over 32 tokens). `--write` saves ops/prices/lfm2.5-24b-a2b/<profile>.<chip key>.json
(the key from tools/gpu, "apple-m5-max"); without it the table is printed.

Why a file at all: with none, the cut learns verify prices of 2+ rows online but never the 1-row price, which
stays at its 1 ms placeholder, so every drafted round looks 15x dearer than the plain step and the lookup
never drafts (measured on the M5 Max: 495 of 499 rounds cut to the anchor on a copy ask).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402

from engine.models.lfm2_moe import mlx_board  # noqa: E402
from engine.models.lfm2_moe.ref import chain  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = {0: 512, 16384: 16384}
ROWS = range(1, 17)


def time_class(board, ctx: int, reps: int) -> dict:
    st = board.new_state(ctx + 64)
    board.prefill(list(range(1000, 1000 + ctx)), st, last_only=True)
    L, conv = st.length, dict(st.conv)
    chain_ms = {}
    for n in ROWS:
        toks = [5 + i for i in range(n)]
        board.rows(toks, chain(n), st)                    # warm the shapes
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            board.rows(toks, chain(n), st)
            ts.append((time.perf_counter() - t0) * 1e3)
        chain_ms[str(n)] = round(statistics.median(ts), 3)
    st.length, st.conv = L, conv
    board.decode(5, st)
    t0 = time.perf_counter()
    for _ in range(32):
        board.decode(5, st)
    plain = (time.perf_counter() - t0) * 1e3 / 32
    return {"chain": chain_ms, "plain": {"1": round(plain, 3)}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--profile", default="lfm25-bf16")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    _, board = mlx_board.load(a.path, log=None)
    doc = {}
    for cls, ctx in CLASSES.items():
        doc[str(cls)] = time_class(board, ctx, a.reps)
        print(f"class {cls}: plain {doc[str(cls)]['plain']['1']} ms, chain {doc[str(cls)]['chain']}", flush=True)
    c0 = doc["0"]["chain"]
    doc["step_ms"] = round((c0["16"] - c0["8"]) / 8, 3)
    doc["measured"] = ("tools/mlx_lfm_prices.py: chain = `Lfm2MoeMLX.rows` of 1..16 chain rows with its eval "
                       f"(host wall clock, the median of {a.reps}), at a 512-token prompt (class 0) and a 16,384-token "
                       "one; plain = the decode step's ms a token over 32 tokens. No spec or prime prior: the cut "
                       "drafts and learns them online.")
    if a.write:
        from engine.methods.device import device_id
        out = os.path.join(ROOT, "ops", "prices", "lfm2.5-24b-a2b", f"{a.profile}.{device_id()}.json")
        with open(out, "w") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")
        print(f"wrote {out}")
    else:
        print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
