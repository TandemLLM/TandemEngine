"""The EXL3 products' launches on a CPU, each read for the order it sums K in (`tools/exl3_rows.Order`).

A CPU cannot reproduce the tensor cores' sums, and on a CPU `exl3_matmul` is the torch reference at every row count,
so the device's class switch is not there to see. `Launches` puts it back without arithmetic: inside it the GPU is
GB10 (`tools/gpu/gb10.py`: every inline extension, its dense rule), every kernel the EXL3 products launch is a
recorder (`fn_dense`, `fn_dense_rows`, `exl3_gemm`, the library GEMM, the slab product `exl3_rows` launches), the
rotations and the slab decode do nothing, and `dev(M, K)` makes an input that answers `is_cuda` (a family's `lin` that
asks takes its device branch). A call's outputs are not computed; what is kept is the order of each product launch,
read from the launch's own arguments: (S, warps) of a dense launch, (S, wk) of the skinny GEMM, (KT, L) or the slice
L of the slab product, the call's shape for the library GEMM (whose order is not ours).

    with Launches() as L:
        o = L.order(lambda: eng.lin(L.dev(M, w.K), w))      # the one order the call's products sum in
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

from tools import exl3_linear as X  # noqa: E402
from tools import exl3_rows as R  # noqa: E402
from tools import fn_kernels as FK  # noqa: E402
from tools import gpu as G  # noqa: E402


class Dev(torch.Tensor):
    """A CPU tensor that a dispatch takes for a device's (its ops keep the type)."""

    @property
    def is_cuda(self):
        return True


class _Jit:
    """A Triton kernel's launch `k[grid](*args, **meta)` (`exl3_rows`' copy of `_slab_rot_s1`), recorded by the
    order its arguments give: chains of L tiles over KT, Q tiles a step."""

    def __init__(self, rec: "Launches"):
        self.rec = rec

    def __getitem__(self, grid):
        def launch(*a, **k):
            KT, L, Q = int(a[6]), int(a[7]), int(k["Q"])
            assert L % Q == 0 and (KT % L) % Q == 0, ("Q must divide every slice", KT, L, Q)
            assert int(a[0].shape[1]) == 16 * KT, ("KT is the input's K tiles", a[0].shape, KT)
            self.rec.add("slab_s1", R.Order.s1(KT, L))
        return launch


class _FakeX:
    """`tools/exl3_linear`'s extension module."""

    def __init__(self, rec):
        self.rec = rec

    def exl3_gemm(self, xs, t, svh, out, kb, cfg, wk, S, b1, b2, chunks):
        self.rec.add("skinny", R.Order.skinny(int(t.shape[0]), int(S), int(wk)))

    def exl3_rot_in(self, x, suh, xh):
        pass

    def exl3_rot_out(self, yp, svh, out):
        pass

    def exl3_recon(self, t, wi, kb, n0):
        pass

    def exl3_rot_in_silu(self, y, suh, xh):
        pass


class _FakeFK:
    """`tools/fn_kernels`' extension module (the dense projections only)."""

    def __init__(self, rec):
        self.rec = rec

    def fn_dense(self, x, t, suh, svh, out, b1, b2, S, warps):
        self.rec.add("dense", R.Order.dense(int(t.shape[0]), int(S), int(warps)))

    def fn_dense_rows(self, x, t, suh, svh, xh, out, b1, b2, S, warps):
        self.rec.add("dense_rows", R.Order.dense(int(t.shape[0]), int(S), int(warps)))


class Launches:
    """The recorder (a context manager). `checks`: how `exl3_rows.check` answers inside it ("orders": the two
    kernels' recorded orders against the prompt order; "fail": the slab unequal)."""

    PRODUCTS = ("dense", "dense_rows", "skinny", "library", "slab_s1")

    def __init__(self, checks: str = "orders"):
        self.calls: list = []
        self.checks = checks
        self.logged: list = []
        self._saved: list = []

    def add(self, kind: str, order) -> None:
        self.calls.append((kind, order))

    def dev(self, M: int, K: int) -> Dev:
        return torch.empty(M, K, dtype=torch.bfloat16).as_subclass(Dev)

    # ---------------------------------------------------------------------------- the patches
    def _set(self, obj, name, val):
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)

    def __enter__(self) -> "Launches":
        gb10 = G.profile_for((12, 1), "NVIDIA GB10")
        rec = self

        def mm(a, b):
            rec.add("library", R.Order.library(a.shape[0], a.shape[1], b.shape[1]))
            return torch.empty(a.shape[0], b.shape[1], dtype=torch.float32)

        def inner(w, n0=0, n1=None):
            n1 = w.N if n1 is None else n1
            return torch.empty(w.K, n1 - n0, dtype=torch.float16)

        def check(w, seed=0):
            want = R.prompt_order(w)
            if self.checks == "fail":
                return {"skinny_equal": True, "slab_equal": False, "one_row_equal": True, "rows": [], "order": str(want)}
            n = max(R.ROWS + 1, R.CROSS + 1)
            mid = self.order(lambda: R.skinny(self.dev(R.ROWS + 1, w.K), w))
            big = self.order(lambda: R.slab(self.dev(n, w.K), w))
            one = self.order(lambda: R.skinny(self.dev(1, w.K), w))
            return {"skinny_equal": mid == want, "slab_equal": big == want, "one_row_equal": one == want,
                    "rows": [R.ROWS + 1, n], "order": str(want)}

        self._set(G, "profile", lambda device=None: gb10)
        self._set(X, "_module", lambda: _FakeX(rec))
        self._set(X, "_mm_fp32", mm)
        self._set(X.EXL3Block, "inner", inner)
        self._set(FK, "_module", lambda: _FakeFK(rec))
        self._set(R, "_kernel", lambda: _Jit(rec))
        self._set(R, "check", check)
        self._set(R, "_BOUND", {})
        self._set(R, "FAILED", {})
        return self

    def __exit__(self, *exc) -> None:
        for obj, name, val in reversed(self._saved):
            setattr(obj, name, val)
        self._saved.clear()

    # ---------------------------------------------------------------------------- reading
    def order(self, fn):
        """The order of the product launches `fn` makes: one Order when they all sum in one order, else the tuple
        of the distinct ones (a call whose launches disagree has no single order)."""
        n0 = len(self.calls)
        fn()
        got = [o for k, o in self.calls[n0:] if k in self.PRODUCTS]
        del self.calls[n0:]
        assert got, "the call launched no product"
        distinct = list(dict.fromkeys(got))
        return distinct[0] if len(distinct) == 1 else tuple(distinct)

    def kinds(self, fn) -> list:
        """The kinds of the product launches `fn` makes, in order."""
        n0 = len(self.calls)
        fn()
        got = [k for k, _ in self.calls[n0:] if k in self.PRODUCTS]
        del self.calls[n0:]
        return got
