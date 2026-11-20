"""LFM2.5 on the Mac's GPU (engine/models/lfm2_moe/mlx_board.py): its verify rows against the decode step on the
real checkpoint, bit for bit.

    .venv-mac/bin/python tools/mlx_lfm_exact.py <checkpoint dir>

A 12-row tree and chains of 16, 5 and 3 rows are verified as one block each after a 300-token prompt; every row
is then recomputed by decoding its path one token at a time from a fresh state, and the two logit rows are
compared with `array_equal`. Exit code 1 when a row differs.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from engine.models.lfm2_moe import mlx_board  # noqa: E402
from engine.models.lfm2_moe.ref import chain  # noqa: E402

BLOCKS = [([7, 19, 23, 44, 31, 60, 88, 91, 5, 17, 300, 4000], [-1, 0, 1, 1, 0, 4, 5, 5, 2, 8, 9, 3]),
          (list(range(50, 66)), chain(16)), (list(range(200, 205)), chain(5)), ([9, 8, 7], chain(3))]


def exact(board) -> tuple[int, int]:
    """(rows compared, rows that differ in any bit) of `BLOCKS` against sequential decode."""
    prompt = list(range(1000, 1300))

    def fresh():
        st = board.new_state(400)
        board.prefill(prompt, st, last_only=True)
        return st

    n = bad = 0
    for tokens, parents in BLOCKS:
        lg = np.array(board.rows(tokens, parents, fresh()))
        for row in range(len(tokens)):
            path, r = [], row
            while r >= 0:
                path.append(r)
                r = parents[r]
            st = fresh()
            for r in reversed(path):
                want = np.array(board.decode(tokens[r], st))
            n += 1
            bad += not np.array_equal(lg[row], want[0])
    return n, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    a = ap.parse_args()
    _, board = mlx_board.load(a.path, log=None)
    n, bad = exact(board)
    print(f"verify rows against sequential decode: {n} rows, {bad} differ in any bit")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
