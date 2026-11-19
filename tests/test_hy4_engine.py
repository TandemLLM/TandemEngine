"""The Hy4-flash engine on a tiny random model, on the CPU: the row path and the prefill against the
reference forward, a verify block's rows against the decode step's (bit for bit), trees, commit of a moved
path, rollback, and the rings, groups and read lists past their first wrap."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.hy4_flash import engine as E  # noqa: E402
from engine.models.hy4_flash.ref import Hy4FlashRef, Hy4State  # noqa: E402
from tests.hy4_tiny import random_weights, tiny_config  # noqa: E402

BOUND = 2e-5      # fp32: the engine's absorbed keys and fp32 sums against the reference's order
g0 = torch.Generator().manual_seed(11)
PROMPT = torch.cat([torch.tensor([5, 9, 3, 61, 22, 2, 3, 40, 41, 12, 63, 7]), torch.randint(4, 60, (33,), generator=g0)])
_T = {}


def tiny(dtype=torch.float32):
    if dtype not in _T:
        cfg = tiny_config()
        W, tables = random_weights(cfg)
        _T[dtype] = (cfg, Hy4FlashRef(cfg, W, tables, dtype=dtype), E.Hy4FlashEngine(cfg, W, tables, dtype=dtype, max_len=128))
    return _T[dtype]


class _small_chunks:
    """Prompt chunks of 7 rows, attention 5 at a time, read lists 3 at a time: every boundary inside 45 rows."""

    def __enter__(self):
        self.old = (E.PREFILL_CHUNK, E.SUB, E.SUB_LIST)
        E.PREFILL_CHUNK, E.SUB, E.SUB_LIST = 7, 5, 3

    def __exit__(self, *exc):
        E.PREFILL_CHUNK, E.SUB, E.SUB_LIST = self.old


def test_prefill_matches_the_reference():
    cfg, ref, eng = tiny()
    want = ref.forward(PROMPT, Hy4State())
    got = eng.prefill(PROMPT, eng.new_state(), head="all")
    err = (got - want).abs().max().item()
    assert err < BOUND, err
    with _small_chunks():
        got2 = eng.prefill(PROMPT, eng.new_state(), head="all")
    err2 = (got2 - want).abs().max().item()
    assert err2 < BOUND, err2
    return f"45 rows (window 4, lists past row 8): one chunk {err:.1e}, chunks of 7 {err2:.1e}"


def test_decode_rows_match_the_reference():
    cfg, ref, eng = tiny()
    want = ref.forward(PROMPT, Hy4State())
    st = eng.new_state()
    got = torch.cat([eng.step(int(t), st) for t in PROMPT])
    err = (got - want).abs().max().item()
    assert st.length == PROMPT.shape[0] and err < BOUND, err
    # a prompt, then steps: the state the prefill leaves is the one the steps continue
    st = eng.new_state()
    with _small_chunks():
        eng.prefill(PROMPT[:19], st, head=None)
    got = torch.cat([eng.step(int(t), st) for t in PROMPT[19:]])
    err2 = (got - want[19:]).abs().max().item()
    assert err2 < BOUND, err2
    return f"45 decode steps {err:.1e}; 19 prompt rows then 26 steps {err2:.1e}"


def _states_equal(a, b) -> bool:
    n = a.length
    return (n == b.length and all(torch.equal(a.lat[k][:n], b.lat[k][:n]) and torch.equal(a.ik[k][:n], b.ik[k][:n]) for k in a.lat)
            and all(torch.equal(a.win[k], b.win[k]) for k in a.win) and torch.equal(a.hist, b.hist))


def test_verify_rows_are_the_decode_steps():
    for dtype in (torch.float32, torch.bfloat16):
        cfg, ref, eng = tiny(dtype)
        res = eng.self_check(PROMPT[:eng.check_rows()].tolist(), steps=24, widths=(2, 3, 4, 8, 24), log=lambda s: None)
        assert res["ok"], res
        # the replay reached the selected groups, wrapped rings, and groups closed inside a block
        cov = res["covers"]
        assert cov["selection"] and cov["rings_wrapped"] and cov["groups_closed_in_blocks"] > 20, cov
        short = eng.self_check(PROMPT[:6].tolist(), steps=4, widths=(2,), log=lambda s: None)
        assert short["ok"] and not short["covers"]["selection"], short
    return f"{res['rows']} rows (chains of 2, 3, 4, 8 and 24, trees of 5) equal the decode steps bit for bit, fp32 and bf16"


def test_commit_a_moved_path_and_a_rollback():
    cfg, ref, eng = tiny(torch.bfloat16)
    a, b = eng.new_state(), eng.new_state()
    eng.prefill(PROMPT[:10], a, head=None)
    eng.prefill(PROMPT[:10], b, head=None)
    t = PROMPT[10:].tolist()
    want = [eng.step(x, a) for x in t[:6]]
    # a tree whose kept path is rows 0, 3, 4 (a rejected branch 1, 2 sits before it), then only two rows of a chain
    got, _ = eng.rows([t[0], 50, 51, t[1], t[2], 52], [-1, 0, 1, 0, 3, 3], b)
    assert all(torch.equal(got[r], want[i][0]) for i, r in enumerate((0, 3, 4)))
    eng.commit(b, [0, 3, 4])
    got, _ = eng.rows([t[3], t[4], 9, 9], [-1, 0, 1, 2], b)
    eng.commit(b, [0, 1])                               # the rollback: two rows kept of four
    assert torch.equal(got[0], want[3][0]) and torch.equal(got[1], want[4][0])
    assert b.length == a.length - 1
    lg = eng.step(t[5], b)
    assert torch.equal(lg, want[5]) and _states_equal(a, b)
    try:
        eng.rows([1, 2], [-1, 0], b)
        eng.commit(b, [1])
        raise SystemExit("a path that skips the anchor was taken")
    except ValueError:
        pass
    return "a tree's moved path and a partly kept chain leave the state of the plain decode, bit for bit"


def test_long_context_wraps_rings_and_selects_groups():
    """120 rows: the 4-row rings wrap 30 times, 60 groups close, every sparse row past row 8 selects."""
    cfg, ref, eng = tiny()
    g = torch.Generator().manual_seed(2)
    toks = torch.randint(4, 60, (120,), generator=g)
    want = ref.forward(toks, Hy4State())
    st = eng.new_state()
    with _small_chunks():
        got = eng.prefill(toks[:90], st, head="all")
    err = (got - want[:90]).abs().max().item()
    assert err < BOUND, err
    blocks, i = [], 90
    while i < 120:
        blk = toks[i:i + 5].tolist()
        lg, _ = eng.rows(blk, list(range(-1, len(blk) - 1)), st)
        eng.commit(st, list(range(len(blk))))
        blocks.append(lg)
        i += 5
    err2 = (torch.cat(blocks) - want[90:]).abs().max().item()
    assert err2 < BOUND and st.length == 120, err2
    return f"90 prompt rows {err:.1e}, then 6 blocks of 5 {err2:.1e}"


def test_a_block_writes_nothing_until_commit():
    cfg, ref, eng = tiny(torch.bfloat16)
    st, twin = eng.new_state(), eng.new_state()
    eng.prefill(PROMPT[:21], st, head=None)
    eng.prefill(PROMPT[:21], twin, head=None)
    eng.rows(PROMPT[21:29].tolist(), list(range(-1, 7)), st)
    assert _states_equal(st, twin)
    return "state unchanged by a block of 8"


def test_the_row_product_reference():
    """`mm`'s contract on its plain form: out[n] = a[n] @ b[n mod NB], any strides, and an item's bits the same in
    any batch (the kernel itself runs on a device: tools/hy4_check.py compares it there)."""
    from engine.models.hy4_flash import kernels as K
    g = torch.Generator().manual_seed(3)
    a, b = torch.randn(12, 5, 70, generator=g), torch.randn(4, 70, 9, generator=g)
    out = K.mm_reference(a, b)
    want = torch.stack([a[n] @ b[n % 4] for n in range(12)])
    assert out.dtype == torch.float32 and torch.equal(out, want)
    assert torch.equal(K.mm_reference(a[4:6], b), out[4:6]) and torch.equal(K.mm_reference(a[:1], b[:1]), out[:1])
    bt = torch.randn(1, 9, 70, generator=g)
    assert torch.equal(K.mm_reference(a, bt.transpose(1, 2)), torch.stack([a[n] @ bt[0].T for n in range(12)]))
    x = torch.randn(3, 70, generator=g).to(torch.bfloat16)
    cfg, ref, eng = tiny(torch.bfloat16)
    assert torch.equal(eng.sumsq(x), torch.stack([(x[n].float() * x[n].float()).sum() for n in range(3)]).view(3, 1)) or \
        (eng.sumsq(x) - x.float().square().sum(-1, keepdim=True)).abs().max() < 1e-3
    assert K.mm_reference(a[:0], b).shape == (0, 5, 9)
    return "items of any batch equal their one-item call; shared, cycled and strided right operands"


def test_capacity_is_checked():
    cfg, ref, eng = tiny()
    st = eng.new_state(12)
    eng.prefill(PROMPT[:11], st, head=None)
    eng.step(5, st)
    for call in (lambda: eng.step(5, st), lambda: eng.prefill(PROMPT[:2], st)):
        try:
            call()
            raise SystemExit("a row past the capacity was taken")
        except ValueError:
            pass
    return "a full state refuses a row"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<52} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<52} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
