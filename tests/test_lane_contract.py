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

The shared code (engine/methods/side_slot.py) answers every pin as each family's copy does:

  5  `ensure_rows` gives every case of 1 (capped at the window for the three that cap, uncapped for
     Flash-Next); under a family's read floor it commits the floor, over the window the window (a window under
     the floor commits the window), inside a capture nothing          test_ensure_rows_*
  6  `side_slot_bytes` is the three families' body; `open_pool` is the five families' line
                                           test_side_slot_bytes_is_the_shared_body, test_open_pool_is_every_familys_line
  7  `lane_drafter` builds the drafter each family's own side slot builds, from the same slot 0 (Flash-Next
     plain, frozen and with a back-off; LFM2.5 with no head)
                                           test_lane_drafter_equals_the_familys_own

The lane contract, through one interface for every family whose host model this file can build (`LANES`:
Flash-Next on its host fake today; Hy4-flash joins with its side slots, the other lane families when their
tiny builders leave their own test files, whose tests stand meanwhile):

  8  a side slot is the family's serving over the same engine (`fn`) with a state of its own and its window;
     a paged side slot takes no bytes a token, and its constant holds at least the sections
                                           test_contract_a_side_slot_shares_the_engine_and_owns_its_state
  9  a family that declares a joint pass has one (`forward_tree_multi`)
                                           test_contract_a_joint_family_has_its_pass
  10 a round of two lanes -- one joint pass, or each lane's own verify in turn -- leaves each lane the logits,
     picks and hidden rows of its lone verify   test_contract_a_round_of_two_lanes_is_each_lanes_lone_verify

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


# ------------------------------------------------------------------ the shared code against the pins

def test_ensure_rows_answers_every_familys_cases():
    from engine.methods.side_slot import ensure_rows
    for name, capped in ENSURE.items():
        def shared(ps, rows, window, capped=capped):
            ensure_rows(ps, rows, window if capped else None)
        for case in ENSURE_CASES:
            assert _ensure_case(shared, case) == _want(case, capped), (name, case)
    ensure_rows(None, 10, 64, floor=40)                 # the pool off: nothing to commit
    return f"{len(ENSURE)} families' {len(ENSURE_CASES)} cases"


def test_ensure_rows_commits_the_floor_and_stops_at_the_window():
    """The design's test 4: rows under the floor commit the floor; rows over the window the window; a window under
    the floor the window; nothing inside a capture."""
    from engine.methods.side_slot import ensure_rows
    base = dict(have=0, margin=8, dev=True, capturing=False)
    cases = [(10, 2048, 4096, 2048), (3000, 2048, 4096, 3000), (5000, 2048, 4096, 4096), (10, 2048, 1024, 1024)]
    for rows, floor, cap, want in cases:
        ps = _Pages(**{k: base[k] for k in ("have", "margin", "dev")})
        real = torch.cuda.is_current_stream_capturing
        torch.cuda.is_current_stream_capturing = lambda: False
        try:
            ensure_rows(ps, rows, cap, floor=floor)
            ensure_rows(ps, rows, cap, floor=floor)                 # committed: one int compare, no call
        finally:
            torch.cuda.is_current_stream_capturing = real
        assert ps.asked == [want], (rows, floor, cap, ps.asked)
    ps = _Pages(0, 8, True)
    real = torch.cuda.is_current_stream_capturing
    torch.cuda.is_current_stream_capturing = lambda: True
    try:
        ensure_rows(ps, 10, 4096, floor=2048)
    finally:
        torch.cuda.is_current_stream_capturing = real
    assert ps.asked == [], "inside a capture: its caller committed the rows"
    return f"{len(cases)} floor and window cases; a capture commits nothing"


def test_side_slot_bytes_is_the_shared_body():
    from engine.methods.side_slot import side_slot_bytes
    cost = CostModel(section_bytes=12_345, view_bytes_per_token=678)
    stub = NS(cost=lambda: cost)
    for name in BYTES_BODY:
        mod, cls = _family(name)
        for paged in (True, False):
            assert side_slot_bytes(cost, paged, mod.SIDE_SLOT_CONST) == cls.side_slot_bytes(stub, 4096, paged), name
    return f"{len(BYTES_BODY)} families"


def test_open_pool_is_every_familys_line():
    from engine.methods.side_slot import open_pool
    for name in POOL_OPEN:
        expr = _own_pool_line(name)
        for a, ctx, opener, on in _pool_cases():
            own = eval(expr, {"getattr": getattr, "hasattr": hasattr}, {"a": a, "ctx": ctx})   # noqa: S307
            assert open_pool(a, ctx) is own is (opener if on else None), (name, vars(a), vars(ctx))
    return f"{len(POOL_OPEN)} families"


def test_lane_drafter_equals_the_familys_own():
    """The design's test 3: from the same slot 0, `lane_drafter` over a new serving and head gives the drafter the
    family's own `side_slot` gives (prices, step, corpus, minimum order, freeze, siblings, back-off). It captures
    the head's graphs once, as Flash-Next's does; LFM2.5's copy calls nothing, and its head (NoHead) captures
    nothing either way."""
    from engine.methods.side_slot import lane_drafter
    from engine.models.flashnext.serving import FlashNextServing
    from engine.models.lfm2_moe.serving import Lfm2MoeServing
    out = []
    for what, lane in (("flashnext", fn_lane), ("flashnext frozen", lambda: fn_lane(freeze=True)),
                       ("flashnext back-off", lambda: fn_lane(backoff=2)), ("lfm2_moe", lfm_lane)):
        s0, d0, _, d2, own_captures = lane()
        if what.startswith("flashnext"):
            e3 = FlashNextServing(s0.fn, 32)
            head = e3.add_mtp_head(d0.depth)
        else:
            e3, head = Lfm2MoeServing(s0.fn, 32), NoHead()
        with _Captures() as cap:
            d3 = lane_drafter(e3, head, d0)
        assert d3.s is e3 and d3.head is head and d3.cut is not d0.cut, what
        assert signature(d3) == signature(d2), (what, signature(d3), signature(d2))
        assert len(cap.calls) == 1 and own_captures == (1 if what.startswith("flashnext") else 0), what
        out.append(what)
    assert NoHead().capture_all() is None
    # a family's own cut over the new prices (Hy4-flash's frozen draft prior goes this way)
    s0, d0, _, _, _ = fn_lane()
    seen = []

    def make_cut(prices, siblings):
        seen.append((prices.table == d0.cut.prices.table, siblings))
        return StairCut(prices, siblings=True)
    with _Captures():
        d4 = lane_drafter(FlashNextServing(s0.fn, 32), NoHead(), d0, make_cut=make_cut)
    assert seen == [(True, False)] and d4.cut.siblings is True
    return ", ".join(out) + "; a family's own cut"


# ------------------------------------------------------------------ the lane contract

class _FnLanes:
    """Flash-Next on the host fake of tests/test_fn_serving.py (its checks at test_fn_serving.py 613 and 649,
    through this file's interface), with an engine pass whose rows depend on the block and the state, never on the
    lane's place in the pass."""

    name = "flashnext"

    def __init__(self):
        from test_fn_serving import _FN
        self.fn = _FN()

        def verify_multi(jobs):
            out = []
            for t, p, st in jobs:
                L = st.length
                out.append((torch.tensor([[float(x + L + j) for j in range(4)] for x in t]),
                            torch.tensor(t) + L, torch.zeros(len(t), 3) + L))
            return out
        self.fn.verify_multi = verify_multi

    def slot0(self, max_len=64):
        from engine.models.flashnext.serving import FlashNextServing
        return FlashNextServing(self.fn, max_len)

    def at(self, s, n):
        s.st.length = n

    def joint(self, items):
        real = torch.cuda.current_stream
        torch.cuda.current_stream = lambda: NS(synchronize=lambda: None)
        try:
            return type(items[0][0]).forward_tree_multi(items)
        finally:
            torch.cuda.current_stream = real


#: the families whose lanes this file builds
LANES = {"flashnext": _FnLanes}


def test_contract_a_side_slot_shares_the_engine_and_owns_its_state():
    for name, F in LANES.items():
        fam = F()
        s0 = fam.slot0()
        assert s0.caps.side_slots, name
        s1, d1 = s0.side_slot(32, None)
        assert type(s1) is type(s0) and s1.fn is s0.fn and s1.st is not s0.st and s1.max_len == 32 and d1 is None, name
        per, const = s0.side_slot_bytes(32, True)
        assert per == 0 and const >= s0.cost().section_bytes, (name, per, const)
        per, _ = s0.side_slot_bytes(32, False)
        assert per >= s0.cost().view_bytes_per_token > 0, (name, per)
    return f"{sorted(LANES)}"


def test_contract_a_joint_family_has_its_pass():
    for name, F in LANES.items():
        s0 = F().slot0()
        if s0.caps.joint_verify:
            assert callable(getattr(type(s0), "forward_tree_multi", None)), f"{name} declares a joint pass and has none"
    return f"{sorted(LANES)}"


def _rows_of(s, lg):
    return lg.clone(), s.picks.clone(), s.hidden_post_norm.clone()


def test_contract_a_round_of_two_lanes_is_each_lanes_lone_verify():
    out = []
    for name, F in LANES.items():
        fam = F()
        blocks = [(torch.tensor([7, 8]), [-1, 0], 5), (torch.tensor([1, 2, 3]), [-1, 0, 0], 9)]

        def pair():
            s0 = fam.slot0()
            s1, _ = s0.side_slot(32, None)
            for s, (_, _, n) in zip((s0, s1), blocks):
                fam.at(s, n)
            return s0, s1
        lanes = pair()
        items = [(s, b, p, n) for s, (b, p, n) in zip(lanes, blocks)]
        if lanes[0].caps.joint_verify:
            together = [_rows_of(s, lg) for s, lg in zip(lanes, fam.joint(items))]
            how = "one joint pass"
        else:
            together = [_rows_of(s, s.forward_tree(b, p, start=n)) for s, b, p, n in items]
            how = "turns"
        for k, (b, p, n) in enumerate(blocks):
            lone = pair()[k]
            if lone.caps.joint_verify:
                ref = _rows_of(lone, fam.joint([(lone, b, p, n)])[0])
            else:
                ref = _rows_of(lone, lone.forward_tree(b, p, start=n))
            for a, r, what in zip(together[k], ref, ("logits", "picks", "hidden")):
                assert torch.equal(a, r), (name, k, what)
        out.append(f"{name}: {how}")
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
