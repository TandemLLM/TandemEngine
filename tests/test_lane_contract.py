"""ENG-485: what every family with side slots builds for a lane, held across families, on the CPU.

Every lane family wrote the same small pieces by copy (docs: the Hy4 lanes design, section 1.7). Before
they are written once in shared code, these pins hold what each family's own copy does today, on its own
code, unchanged:

  1  `_ensure`: the KV pool's pages for rows [0, rows) committed before a write, never inside a capture;
     Flash-Next's has no cap, LFM2.5's, Flint's and NemotronH's cap the rows at the window
                                           test_pin_every_familys_ensure
  2  `side_slot_bytes`: (0 a token when paged, else the view bytes a token; the sections and the family's
     constant), the body LFM2.5, Flint and NemotronH share (Flash-Next adds its MTP head's cache)
                                           test_pin_the_side_slot_bytes_body
  3  the pool-open line of `serve`: the pool when `--kv-pool on` and the loader offers one, else None
     (Flash-Next, LFM2.5, Flint, NemotronH, and the Qwen3.5 MoE trunk)
                                           test_pin_every_familys_pool_open_line
  4  a side slot's drafter: a new price model over slot 0's table and step, a lookup over slot 0's corpus
     (min order 3 as slot 0's, depth 16, corpus order 8), slot 0's freeze and lookup back-off; checked on
     the host fakes of Flash-Next (tests/test_fn_serving.py) and LFM2.5 (tests/test_lfm2_moe_serving.py)
                                           test_pin_a_side_slots_drafter

Run: python tests/test_lane_contract.py
"""

from __future__ import annotations

import inspect
import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import test_forward_tree  # noqa: E402,F401  (the CPU environment before the engine imports)
import torch  # noqa: E402

from engine.drafters.ngram import NgramDrafter  # noqa: E402
from engine.methods.mtp_drafter import MTPTreeDrafter, NoHead  # noqa: E402
from engine.methods.staircut import PriceModel, StairCut  # noqa: E402
from engine.models.adapter import CostModel  # noqa: E402

NS = types.SimpleNamespace


def _family(name: str):
    """A family's serving module and its serving class."""
    import importlib
    mod = importlib.import_module(f"engine.models.{name}.serving")
    cls = {"flashnext": "FlashNextServing", "lfm2_moe": "Lfm2MoeServing", "flint": "FlintServing",
           "nemotron_h": "NemotronHServing", "qwen3_5_moe": None}[name]
    return mod, (getattr(mod, cls) if cls else None)


# ------------------------------------------------------------------ 1: _ensure

#: the families whose `_ensure` is lifted, and whether theirs caps the rows at the window
ENSURE = {"flashnext": False, "lfm2_moe": True, "flint": True, "nemotron_h": True}

#: (rows asked, rows committed, margin, window, pages on the device, a capture running) -> rows `ensure` gets
#: (None: no call); `cap` marks the case a capped copy answers otherwise
ENSURE_CASES = [
    dict(rows=10, have=0, margin=8, window=64, dev=False, capturing=False, want=10),
    dict(rows=10, have=18, margin=8, window=64, dev=False, capturing=False, want=None),     # committed with the margin
    dict(rows=11, have=18, margin=8, window=64, dev=False, capturing=False, want=11),
    dict(rows=100, have=0, margin=8, window=64, dev=False, capturing=False, want=100, cap=64),
    dict(rows=64, have=72, margin=8, window=64, dev=False, capturing=False, want=None),
    dict(rows=10, have=0, margin=8, window=64, dev=True, capturing=True, want=None),        # inside a capture
    dict(rows=10, have=0, margin=8, window=64, dev=False, capturing=True, want=10),         # host pages: no capture
    dict(rows=10, have=0, margin=8, window=64, dev=True, capturing=False, want=10),
]


class _Pages:
    """A PagedSet's bookkeeping: its committed rows, its margin, where it lives, what `ensure` was asked."""

    def __init__(self, have: int, margin: int, dev: bool):
        self.rows, self._m, self._dev, self.asked = have, margin, dev, []

    def margin(self) -> int:
        return self._m

    def on_device(self) -> bool:
        return self._dev

    def ensure(self, rows: int) -> None:
        self.asked.append(int(rows))
        self.rows = int(rows) + self._m


def _ensure_case(fn, case) -> list:
    """`fn(pages, rows, window)` for one case with the capture flag patched; the rows `ensure` was asked."""
    ps = _Pages(case["have"], case["margin"], case["dev"])
    real = torch.cuda.is_current_stream_capturing
    torch.cuda.is_current_stream_capturing = lambda: case["capturing"]
    try:
        fn(ps, case["rows"], case["window"])
    finally:
        torch.cuda.is_current_stream_capturing = real
    return ps.asked


def _own_ensure(cls):
    def run(ps, rows, window):
        cls._ensure(NS(st=NS(pages=ps), max_len=window), rows)
    return run


def _want(case, capped: bool) -> list:
    w = case.get("cap", case["want"]) if capped else case["want"]
    return [] if w is None else [w]


def test_pin_every_familys_ensure():
    for name, capped in ENSURE.items():
        _, cls = _family(name)
        for case in ENSURE_CASES:
            assert _ensure_case(_own_ensure(cls), case) == _want(case, capped), (name, case)
        # no pages (the pool off): nothing to commit
        cls._ensure(NS(st=NS(pages=None), max_len=64), 10)
    return f"{len(ENSURE)} families x {len(ENSURE_CASES)} cases; Flash-Next's alone uncapped"


# ------------------------------------------------------------------ 2: side_slot_bytes

#: the families whose `side_slot_bytes` body is the shared one
BYTES_BODY = ("lfm2_moe", "flint", "nemotron_h")


def test_pin_the_side_slot_bytes_body():
    cost = CostModel(section_bytes=12_345, view_bytes_per_token=678)
    stub = NS(cost=lambda: cost)
    for name in BYTES_BODY:
        mod, cls = _family(name)
        for paged in (True, False):
            got = cls.side_slot_bytes(stub, 4096, paged)
            assert got == ((0 if paged else 678), 12_345 + mod.SIDE_SLOT_CONST), (name, paged, got)
    return f"{len(BYTES_BODY)} families: 0 or the view bytes a token, the sections and the family's constant"


# ------------------------------------------------------------------ 3: the pool-open line

#: the families whose `serve` opens the pool with the one line
POOL_OPEN = ("flashnext", "lfm2_moe", "flint", "nemotron_h", "qwen3_5_moe")
POOL_LINE = re.compile(r"^\s*pool = (ctx\.kv_pool if .*)$", re.M)


def _pool_cases():
    """(a, ctx, whether the loader's opener is the answer)"""
    opener = object()
    for kv in ("on", "off", None):
        for has in (True, False):
            a = NS() if kv is None else NS(kv_pool=kv)
            ctx = NS(kv_pool=opener) if has else NS()
            yield a, ctx, opener, kv == "on" and has


def _own_pool_line(name: str) -> str:
    mod, _ = _family(name)
    found = POOL_LINE.findall(inspect.getsource(mod.serve))
    assert len(found) == 1, (name, found)
    return found[0]


def test_pin_every_familys_pool_open_line():
    for name in POOL_OPEN:
        expr = _own_pool_line(name)
        for a, ctx, opener, on in _pool_cases():
            got = eval(expr, {"getattr": getattr, "hasattr": hasattr}, {"a": a, "ctx": ctx})   # noqa: S307
            assert got is (opener if on else None), (name, vars(a), vars(ctx))
    return f"{len(POOL_OPEN)} families' line: the opener only with --kv-pool on and a loader that offers one"


# ------------------------------------------------------------------ 4: a side slot's drafter

def _drafter0(s0, head, siblings: bool, freeze: bool = False, backoff: int = 0):
    """Slot 0's drafter as its family's builder makes it: a price file's table and step, a lookup of min order 3."""
    ng0 = NgramDrafter(corpus_path="", min_order=3, max_depth=16, min_corpus_order=8)
    ng0.corpus = object()                               # the server's suffix store, by identity
    d0 = MTPTreeDrafter(s0, head, StairCut(PriceModel({"0": {"chain": {"1": 20.0}}, "4096": {"chain": {"3": 31.0}}},
                                                      2.0), siblings=siblings), ng0, freeze=freeze)
    d0.mtp_look_backoff = backoff
    return d0


def signature(d) -> dict:
    """What a side slot's drafter takes from slot 0's: its cut's kind, prices and switches, its lookup, its
    back-offs."""
    c, ng = d.cut, d.ngram
    return {"cut": type(c).__name__, "table": c.prices.table, "step": c.prices.step, "siblings": c.siblings,
            "frozen": c.frozen, "freeze": d.freeze, "mtp_look_backoff": d.mtp_look_backoff,
            "look_backoff": d.look_backoff, "anchor_backoff": d.anchor_backoff,
            "ngram": None if ng is None else {"corpus": id(ng.corpus), "min_order": ng.min_order,
                                               "max_depth": ng.max_depth, "min_corpus_order": ng.min_corpus_order}}


def lane_rule(d0) -> dict:
    """The drafter a side slot gets from slot 0's `d0`, as the families build it today."""
    c0, ng0 = d0.cut, d0.ngram
    return {"cut": "StairCut", "table": c0.prices.table, "step": c0.prices.step, "siblings": c0.siblings,
            "frozen": bool(d0.freeze), "freeze": bool(d0.freeze), "mtp_look_backoff": d0.mtp_look_backoff,
            "look_backoff": 0, "anchor_backoff": 0,
            "ngram": None if ng0 is None else {"corpus": id(ng0.corpus), "min_order": ng0.min_order,
                                                "max_depth": 16, "min_corpus_order": 8}}


class _Captures:
    """MTPTreeDrafter.capture_all counted, not run (a host fake has no graphs)."""

    def __enter__(self):
        self.calls, self.real = [], MTPTreeDrafter.capture_all
        MTPTreeDrafter.capture_all = lambda d: self.calls.append(d)
        return self

    def __exit__(self, *exc):
        MTPTreeDrafter.capture_all = self.real
        return False


def fn_lane(freeze=False, backoff=0):
    """Flash-Next on the host fake of tests/test_fn_serving.py: slot 0, its drafter, and its own side slot's."""
    from engine.models.flashnext.serving import FlashNextServing
    from test_fn_serving import _FN
    s0 = FlashNextServing(_FN(), 64)
    d0 = _drafter0(s0, s0.add_mtp_head(3), siblings=False, freeze=freeze, backoff=backoff)
    with _Captures() as cap:
        e2, d2 = s0.side_slot(32, d0)
    return s0, d0, e2, d2, len(cap.calls)


def lfm_lane():
    """LFM2.5 on the host fake of tests/test_lfm2_moe_serving.py, the same."""
    from engine.models.lfm2_moe.serving import Lfm2MoeServing
    from test_lfm2_moe_serving import _Eng
    s0 = Lfm2MoeServing(_Eng(), 64)
    d0 = _drafter0(s0, NoHead(), siblings=False)
    with _Captures() as cap:
        e2, d2 = s0.side_slot(32, d0)
    return s0, d0, e2, d2, len(cap.calls)


def test_pin_a_side_slots_drafter():
    out = []
    for what, (s0, d0, e2, d2, captured) in (("flashnext", fn_lane()), ("flashnext frozen", fn_lane(freeze=True)),
                                             ("flashnext back-off", fn_lane(backoff=2)), ("lfm2_moe", lfm_lane())):
        assert d2.s is e2 and e2 is not s0 and d2.cut is not d0.cut and d2.cut.prices is not d0.cut.prices, what
        assert d2.ngram is not d0.ngram and d2.ngram.corpus is d0.ngram.corpus, what
        assert signature(d2) == lane_rule(d0), (what, signature(d2), lane_rule(d0))
        out.append(f"{what}: captured {captured}")
    # what differs today: Flash-Next captures the lane's head graphs at once (1 call), LFM2.5 calls nothing
    # (its head is NoHead, whose capture is a no-op)
    return "; ".join(out)


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<58} ok   {fn() or ''}")
        except Exception as exc:                                   # noqa: BLE001
            fails += 1
            import traceback
            traceback.print_exc()
            print(f"  {name:<58} FAIL ({exc})")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
