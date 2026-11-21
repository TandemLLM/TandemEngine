"""tools/exl3_rows.py on a CPU (R83): the orders it reads are the kernels' own, and its dispatch launches one.

  1  every slice has one length: the prompt order (S1 w8, exl3.cu's statements read from the source and run) for
     every shape of the four split tables and the device's rule, and `dense`'s order (kernels.cu's statements: RowGemm's
     warp slices, the rank slices of `dense_body` and `dense_rows_kernel`, the binding's check) at every table split
     and the device's; each is one chain of equal slices, so `_slab_rot_s1` runs it with Q dividing L (Hy4's
     kv_a_proj_with_mqa on the rule, fn 640x2560, lfm 2048x11776 step two tiles)
                                                                   test_every_slice_has_one_length
  2  the skinny slices are exl3.cu's and `exl3_slab_gemm.skinny_slices`' at `pick`'s split
                                                                   test_the_skinny_slices_are_the_kernels_formulas
  3  equality is the chain: empty slices and singleton groups fold; a flat chain is not a ranked one; the library
     product equals no order of ours                               test_equal_orders_are_equal_chains
  4  an order's s1 launch reads back as the order                  test_an_s1_launch_reads_back_as_its_order
  5  `linear`: the skinny GEMM at S1 w8 to CROSS rows (in EXL3_CHUNK_UNTIL-row calls past that), then the slab's s1
     at L = KT / 8, one order; a half-integer rate on the slab at every row count; a check that is unequal, or a rate
     the kernels do not decode, raises (FAILED keeps the shape); off the kernels exl3_matmul
                                                                   test_the_dispatch
  6  one check a shape and device                                   test_one_check_a_shape

CPU, no kernels (tests/exl3_launches.py records the launches). Run: python tests/test_exl3_rows.py
"""

from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import torch  # noqa: E402

from exl3_launches import Launches  # noqa: E402
from tools import exl3_linear as X  # noqa: E402
from tools import exl3_rows as R  # noqa: E402
from tools.gpu.gb10 import GB10  # noqa: E402

TABLES = ("fn-dense-splits.json", "lfm25-dense-splits.json", "flint-dense-splits.json", "hy4-dense-splits.json")
#: Hy4's kv_a_proj_with_mqa (576 columns stored as 640, Kin 3,072): no table names it, the rule gives S8 w4
KV_A = (640, 3072, 5)


def _src(*parts) -> str:
    with open(os.path.join(ROOT, *parts)) as f:
        return f.read()


def _stmts(src: str, *lines: str) -> str:
    """The C statements `lines` (each must be in `src` verbatim) as Python: declarations dropped, integer division."""
    out = []
    for ln in lines:
        assert ln in src, f"kernel source changed: {ln!r} is gone (the order is read from it)"
        for s in ln.split(";"):
            s = s.strip().replace("const int ", "")
            if s:
                out += [p.replace(" / ", " // ") for p in re.split(r",\s*(?=[a-z_0-9]+ = )", s)]
    return "\n".join(out)


DENSE_RANK = ("const int KT = Kin / 16, per_r = KT / S;", "const int ka = rank * per_r, kb = ka + per_r;")
DENSE_WARP = ("const int per = (kb - ka + WARPS - 1) / WARPS;", "k0 = min(kb, ka + warp * per); k1 = min(kb, k0 + per);")
DENSE_BIND = "(Kin / 16) % (8 * S) == 0"
SKINNY = ("const int per_r = (KT + S - 1) / S;", "const int r0 = min(KT, rank * per_r), r1 = min(KT, r0 + per_r);",
          "const int per_w = (r1 - r0 + WK - 1) / WK;", "const int k0 = min(r1, r0 + wk * per_w), k1 = min(r1, k0 + per_w);")


def _cuda_dense(Kin: int, S: int, W: int) -> list:
    src = _src("tools", "csrc", "fn_kernels", "kernels.cu")
    assert src.count(DENSE_RANK[0]) >= 2 and src.count(DENSE_RANK[1]) >= 2, "dense_body / dense_rows_kernel"
    assert src.count(DENSE_BIND) >= 2, "fn_dense / fn_dense_rows' split check"
    rank_c, warp_c = _stmts(src, *DENSE_RANK), _stmts(src, *DENSE_WARP)
    sl = []
    for rank in range(S):
        env = {"Kin": Kin, "S": S, "rank": rank}
        exec(rank_c, {}, env)
        for warp in range(W):
            e = dict(env, WARPS=W, warp=warp)
            exec(warp_c, {"min": min}, e)
            sl.append((e["k0"], e["k1"]))
    return sl


def _cuda_skinny(Kin: int, S: int, WK: int) -> list:
    code = _stmts(_src("tools", "csrc", "exl3_linear", "exl3.cu"), *SKINNY)
    groups = []
    for rank in range(S):
        g = []
        for wk in range(WK):
            e = {"KT": Kin // 16, "S": S, "WK": WK, "rank": rank, "wk": wk}
            exec(code, {"min": min}, e)
            g.append((e["k0"], e["k1"]))
        groups.append(g)
    return groups


def _table_shapes() -> list:
    out = []
    for t in TABLES:
        with open(os.path.join(ROOT, "ops", t)) as f:
            for k, v in json.load(f).items():
                if not k.startswith("_"):
                    N, K, kb = (int(x) for x in k.split("x"))
                    out.append((t, N, K, kb, int(v["S"]), int(v["warps"])))
    return out


def _equal_slices(sl: list, KT: int, n: int) -> int:
    L = KT // n
    assert KT % n == 0 and sl == [(i * L, (i + 1) * L) for i in range(n)], (KT, n, sl)
    return L


def test_every_slice_has_one_length():
    rule = GB10((12, 1), "GB10")
    cases = _table_shapes()
    cases += [("rule", N, K, kb, *rule.dense_split(N, K, kb, {})) for _, N, K, kb, _, _ in list(cases) + [("", *KV_A, 0, 0)]]
    qs = {"prompt": set(), "dense": set()}
    for src, N, K, kb, S, W in cases:
        KT = K // 16
        # the prompt order: the skinny GEMM at S1 w8
        p = _cuda_skinny(K, R.S_PROMPT, R.WK_PROMPT)
        L = _equal_slices(p[0], KT, R.WK_PROMPT)
        o = R.prompt_order(X.EXL3Block(torch.zeros(KT, N // 16, X.tile_width(kb), dtype=torch.int16),
                                       torch.ones(K, dtype=torch.float16), torch.ones(N, dtype=torch.float16)))
        assert o == R.Order.chain(p) == R.Order.s1(KT, L)
        kt, l, q = o.s1_launch()
        assert (kt, l) == (KT, L) and L % q == 0 and q == next(v for v in (4, 2, 1) if L % v == 0)
        qs["prompt"].add(q)
        # dense's order at the split
        assert KT % (8 * S) == 0 and W in (4, 8), (src, N, K, kb, S, W)
        d = _cuda_dense(K, S, W)
        L = _equal_slices(d, KT, S * W)
        od = R.Order.dense(KT, S, W)
        assert od == R.Order.chain([d]) == R.Order.s1(KT, L)
        kt, l, q = od.s1_launch()
        assert (kt, l) == (KT, L) and L % q == 0 and q == next(v for v in (4, 2, 1) if L % v == 0)
        qs["dense"].add(q)
    # the 32-wide K steps of the later families' X pair
    assert rule.dense_split(*KV_A, {}) == (8, 4) and R.Order.dense(192, 8, 4).s1_launch() == (192, 6, 2)
    assert R.Order.dense(160, 4, 4).s1_launch() == (160, 10, 2)
    assert R.Order.dense(736, 2, 8).s1_launch() == (736, 46, 2)
    assert qs["prompt"] == {4} and qs["dense"] >= {2, 4}, qs
    return f"{len(cases)} (shape, split) cases: one slice length each; prompt steps {sorted(qs['prompt'])}, " \
           f"dense steps {sorted(qs['dense'])}"


def test_the_skinny_slices_are_the_kernels_formulas():
    from tools import exl3_slab_gemm as SG
    n = 0
    for _, N, K, kb, _, _ in _table_shapes():
        if kb >= 10:
            continue
        t = X.pick(N, K, kb)
        S, WK = int(t["S"]), int(t["wk"])
        want = _cuda_skinny(K, S, WK)
        assert R.Order.skinny(K // 16, S, WK) == R.Order.chain(want), (N, K, kb)
        S2, WK2, sl, q = SG.skinny_slices(N, K, kb, "cpu")
        assert (S2, WK2) == (S, WK) and [tuple(x) for x in sl.view(-1, 2).tolist()] == [s for g in want for s in g]
        assert R.Order.sliced(sl, S, WK) == R.Order.skinny(K // 16, S, WK)
        n += 1
    return f"{n} shapes: exl3.cu's slices and skinny_slices' bounds"


def test_equal_orders_are_equal_chains():
    O = R.Order
    # a flat chain is a flat chain whichever kernel sums it: Hy4's 8192x3072 decode split is its prompt order
    assert O.dense(192, 1, 8) == O.skinny(192, 1, 8) == O.s1(192, 24)
    assert O.dense(192, 1, 4) != O.skinny(192, 1, 8)
    # ranks added as groups are not one chain: dense S2 w8 (flat, 16 slices) against skinny S2 wk8 (two groups of 8)
    a, b = O.dense(1024, 2, 8), O.skinny(1024, 2, 8)
    assert [s for g in b.key for s in g] == list(a.key[0]) and a != b
    # nor is S2 w8's flat chain of 16 slices the prompt order's chain of 8
    assert a != O.skinny(1024, 1, 8)
    # an empty slice adds +0 and goes; a group of one slice after the first is the first chain's next slice
    assert O.chain([[(0, 4), (4, 4), (4, 8)]]) == O.chain([[(0, 4), (4, 8)]])
    assert O.chain([[(0, 4), (4, 8)], [(8, 12)]]) == O.chain([[(0, 4), (4, 8), (8, 12)]])
    assert O.chain([[(0, 4)], [(4, 8), (8, 12)]]) != O.chain([[(0, 4), (4, 8), (8, 12)]])
    # skinny with one tile a rank: each rank a single slice (its other warps empty), so a flat chain
    assert O.skinny(2, 2, 8) == O.s1(2, 1)
    lib = O.library(211, 3072, 1536)
    assert not lib.ours and lib != O.library(212, 3072, 1536) and lib == O.library(211, 3072, 1536)
    assert all(lib != o for o in (a, b, O.s1(192, 24)))
    return "chains compare by their partials' order, not their launch"


def test_an_s1_launch_reads_back_as_its_order():
    for KT, S, W in ((1024, 2, 8), (192, 1, 8), (160, 4, 4), (736, 2, 8), (192, 8, 4), (512, 2, 8)):
        o = R.Order.dense(KT, S, W)
        kt, L, Q = o.s1_launch()
        assert R.Order.s1(kt, L) == o and L % Q == 0
    for KT in (96, 192, 512, 1024, 40, 24):
        o = R.Order.skinny(KT, R.S_PROMPT, R.WK_PROMPT)
        kt, L, Q = o.s1_launch()
        assert R.Order.s1(kt, L) == o and (kt, L) == (KT, -(-KT // 8)) and L % Q == 0 and (kt % L) % Q == 0
    assert R.Order.skinny(1024, 2, 8).s1_launch() is None
    assert R.Order.library(5, 128, 128).s1_launch() is None
    return "every s1 launch reads back as its order; a ranked chain and the library have none"


def _block(N: int, K: int, kb: int) -> X.EXL3Block:
    t = torch.zeros(K // 16, N // 16, X.tile_width(kb), dtype=torch.int16)
    return X.EXL3Block(t, torch.ones(K, dtype=torch.float16), torch.ones(N, dtype=torch.float16))


def test_the_dispatch():
    with Launches() as L:
        for N, K, kb in ((3072, 16384, 5), (8192, 3072, 5), (1536, 3072, 5), KV_A, (2048, 1024, 35)):
            w = _block(N, K, kb)
            want = R.Order.s1(K // 16, K // 16 // 8)
            assert R.prompt_order(w) == want
            rows = ((1, ["skinny"]), (2, ["skinny"]), (32, ["skinny"]), (33, None), (128, None), (129, None),
                    (R.CROSS, None), (R.CROSS + 1, ["slab_s1"]), (4096, ["slab_s1"]))
            if kb >= 10:        # a half-integer rate: the slab at every row count
                rows = tuple((M, ["slab_s1"]) for M, _ in rows)
            for M, kinds in rows:
                got = L.kinds(lambda: R.linear(L.dev(M, w.K), w))
                assert (kinds is None and got and set(got) == {"skinny"}) or got == kinds, ((N, K, kb), M, got)
                assert L.order(lambda: R.linear(L.dev(M, w.K), w)) == want, ((N, K, kb), M)
            # past EXL3_CHUNK_UNTIL the skinny GEMM runs in calls of that many rows
            if kb < 10:
                assert len(L.kinds(lambda: R.linear(L.dev(R.CROSS, w.K), w))) >= R.CROSS // X.CHUNK_UNTIL
            # the skinny GEMM's own split is not this order on S2 shapes: the order is passed, not read
            if kb < 10 and X.pick(N, K, kb)["S"] != 1:
                assert L.order(lambda: X.exl3_matmul(L.dev(40, K), w)) != want
    with Launches(checks="fail") as L:
        w = _block(3072, 8192, 5)
        try:
            R.linear(L.dev(200, w.K), w)
        except R.RowOrderError as e:
            assert "3072x8192" in str(e)
        else:
            raise AssertionError("an unequal check must fail the call")
        assert len(R.FAILED) == 1 and not next(iter(R.FAILED.values()))["slab_equal"]
        odd = _block(2048, 1024, 5)
        odd.kb = 9                      # no rate the kernels decode
        try:
            R.linear(L.dev(200, odd.K), odd)
        except R.RowOrderError as e:
            assert "rate" in str(e)
        else:
            raise AssertionError("an unknown rate has no prompt order: RowOrderError")
    # off the kernels (this CPU): exl3_matmul, the torch reference
    x = (torch.randn(3, 256) * 0.5).to(torch.bfloat16)
    b = X.random_block(256, 256, 4, device="cpu", seed=1)
    assert torch.equal(R.linear(x, b), X.exl3_matmul(x, b))
    return f"skinny S1 w8 to CROSS {R.CROSS} (calls of {X.CHUNK_UNTIL} past it), s1 above, a 3.5-bit rate s1 at every " \
           f"row count, one order; an unequal check or an unknown rate raises; the CPU path unchanged"


def test_one_check_a_shape():
    w = _block(1536, 3072, 5)
    with Launches() as L:
        calls = []
        fake = R.check
        R.check = lambda *a, **k: (calls.append(1), fake(*a, **k))[1]
        try:
            for M in (1, 20, 40, 200, 4096):
                R.linear(L.dev(M, w.K), w)
        finally:
            R.check = fake
        assert len(calls) == 1, calls
    return "the check runs once a shape, at its first call past 32 rows"


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
