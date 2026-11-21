"""The linear op's order at every row count of a prompt pass (R83; a guard on the dispatch, not evidence of bits: the
GPU guard tests/gpu/test_linear_rowinv_gpu.py is that). The order the product's launches sum K in, from 1 to 4,097
rows, against the one-row call's.

A CPU cannot see this with arithmetic: `exl3_matmul` is the torch reference there at every row count, and the tensor
cores' sums are not a CPU's. So the dispatch runs against tests/exl3_launches.py, which takes the device's branch and
reads the order of each launch from its arguments (dense: S and warps; the skinny GEMM: S and wk; the slab product:
its chain of L tiles; the library GEMM: not ours).

  1  the prompt order (tools/exl3_rows.linear) is one order at every row count, on every shape of the four split
     tables (the half-integer ones on the slab at every row count), Hy4's kv_a_proj_with_mqa and its 3.5-bit indexer
     query projection, and a shape on the device's rule  test_the_prompt_dispatch_has_one_order
  2  two-sided, today: every family's `lin` in TODAY switches its order by rows, first at the row count recorded
     (33 and 129, 34 on NemotronH). A family whose `lin` takes one order for its prompt rows deletes its line
                                                                 test_today_every_familys_lin_switches_order_by_rows

The shapes: each family's split table (as its loader applies it, with its tile table where it has one) and a shape
on the device's rule; the vocabulary heads are left out (a prompt reads the head on its last row). The 27B is not
here: its prompt pass is out of R83 by the owner's ruling (the rowinv design, section 3.3).

CPU, no kernels. Run: python tests/test_linear_order.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import torch  # noqa: E402

from exl3_launches import Launches  # noqa: E402
from tools import exl3_linear as X  # noqa: E402
from tools import exl3_rows as R  # noqa: E402
from tools import fn_kernels as FK  # noqa: E402

#: every row count the guard asks for on a family's first shape; its other shapes take G1's list and both sides of
#: every class line
ALL_ROWS = range(1, 4098)
SOME_ROWS = sorted({1, 2, 15, 16, 17, 31, 32, 33, 34, 35, 64, 74, 127, 128, 129, 130, 211, 255, 256, 257, 349, 423,
                    511, 512, 513, 1024, 4095, 4096, 4097})
#: a shape on the device's rule (no table names it)
RULE_SHAPE = (4096, 4096, 4)
#: heads past this many columns are left out
MAX_N = 65536

#: this tree: the first row count at which a family's `lin` sums some shape in another order than its one-row call
TODAY = {"hy4_flash": 33, "flashnext": 129, "qwen3_5_moe": 129, "lfm2_moe": 33, "flint": 33, "nemotron_h": 34}


def _block(N: int, K: int, kb: int) -> X.EXL3Block:
    """A block of the shape (the trellis' bits are never read here)."""
    t = torch.zeros(K // 16, N // 16, X.tile_width(kb), dtype=torch.int16)
    return X.EXL3Block(t, torch.ones(K, dtype=torch.float16), torch.ones(N, dtype=torch.float16))


def _table(name: str) -> list:
    with open(os.path.join(ROOT, "ops", name)) as f:
        return [tuple(int(x) for x in k.split("x")) for k in json.load(f) if not k.startswith("_")]


class _Tables:
    """The family's tables applied as its loader applies them, and taken back after."""

    def __init__(self, apply):
        self.apply = apply

    def __enter__(self):
        self.dense, self.cfg = dict(FK.DENSE_SPLIT), dict(X._CONFIG)
        self.apply()

    def __exit__(self, *exc):
        FK.DENSE_SPLIT.clear()
        FK.DENSE_SPLIT.update(self.dense)
        X._CONFIG.clear()
        X._CONFIG.update(self.cfg)


def _bare(cls, **attrs):
    """An engine object without its weights: `lin` reads only these."""
    o = object.__new__(cls)
    for k, v in attrs.items():
        setattr(o, k, v)
    return o


def families() -> dict:
    """name -> (lin(x, w), its tables' context, its shapes)."""
    out = {}

    def shapes(*tables):
        s = [t for name in tables for t in _table(name)] + [RULE_SHAPE]
        return [t for t in dict.fromkeys(s) if t[0] <= MAX_N]

    from engine.models.hy4_flash.engine import Hy4FlashEngine
    from engine.models.hy4_flash.load import apply_splits as hy4_splits
    e = _bare(Hy4FlashEngine, kernels=True, block_rows=True, cuts={})
    out["hy4_flash"] = (e.lin, _Tables(hy4_splits), shapes("hy4-dense-splits.json"))

    from engine.models.flashnext.engine import FlashNextEngine
    from engine.models.flashnext.load import apply_tiles
    fn = shapes("fn-dense-splits.json")
    out["flashnext"] = (_bare(FlashNextEngine).lin, _Tables(apply_tiles), fn)
    from engine.models.qwen3_5_moe.engine import Qwen35MoeEngine
    out["qwen3_5_moe"] = (_bare(Qwen35MoeEngine).lin, _Tables(apply_tiles), fn)

    from engine.models.lfm2_moe.engine import Lfm2MoeEngine
    from engine.models.lfm2_moe.load import apply_splits as lfm_splits
    out["lfm2_moe"] = (_bare(Lfm2MoeEngine).lin, _Tables(lfm_splits), shapes("lfm25-dense-splits.json"))

    from engine.models.flint.engine import FlintEngine
    from engine.models.flint.load import apply_splits as flint_splits
    out["flint"] = (_bare(FlintEngine).lin, _Tables(flint_splits), shapes("flint-dense-splits.json"))

    from engine.models.nemotron_h.engine import NemotronHEngine
    out["nemotron_h"] = (_bare(NemotronHEngine).lin, _Tables(lambda: None), [RULE_SHAPE, (2688, 2688, 4), (5376, 2688, 4)])
    return out


def first_break(L: Launches, lin, w, rows) -> tuple:
    """(the one-row call's order, the first row count whose call sums in another one or None)."""
    one = L.order(lambda: lin(L.dev(1, w.K), w))
    for M in rows:
        if L.order(lambda: lin(L.dev(M, w.K), w)) != one:
            return one, M
    return one, None


def _survey(lin, shapes) -> dict:
    got = {}
    with Launches() as L:
        for i, (N, K, kb) in enumerate(shapes):
            w = _block(N, K, kb)
            one, at = first_break(L, lin, w, ALL_ROWS if i == 0 else SOME_ROWS)
            got[f"{N}x{K}x{kb}"] = (str(one), at)
    return got


def test_the_prompt_dispatch_has_one_order():
    shapes = [s for name in ("hy4-dense-splits.json", "fn-dense-splits.json", "lfm25-dense-splits.json",
                             "flint-dense-splits.json") for s in _table(name) if s[0] <= MAX_N]
    shapes = list(dict.fromkeys(shapes + [(640, 3072, 5), (2048, 1024, 35), RULE_SHAPE]))
    with Launches() as L:
        for i, (N, K, kb) in enumerate(shapes):
            w = _block(N, K, kb)
            one, at = first_break(L, R.linear, w, ALL_ROWS if i == 0 else SOME_ROWS)
            assert at is None, ((N, K, kb), str(one), at)
            assert one == R.prompt_order(w) == R.Order.s1(K // 16, K // 128), ((N, K, kb), str(one))
    return f"{len(shapes)} shapes: one order at every row count"


def test_today_every_familys_lin_switches_order_by_rows():
    fams = families()
    assert set(fams) == set(TODAY), (sorted(fams), sorted(TODAY))
    lines = []
    for name, (lin, tables, shapes) in fams.items():
        with tables:
            got = _survey(lin, shapes)
        breaks = {k: at for k, (_, at) in got.items()}
        one = sorted(k for k, at in breaks.items() if at is None)
        assert not one, f"{name}: {one} sum in one order at every row count: if its lin now takes one order for " \
                        f"prompt rows, delete its line in TODAY"
        first = min(breaks.values())
        assert first == TODAY[name], (name, first, TODAY[name], breaks)
        lines.append(f"{name} {first} ({len(shapes)} shapes)")
    return "; ".join(lines)


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
