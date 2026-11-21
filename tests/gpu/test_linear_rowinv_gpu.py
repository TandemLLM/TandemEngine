"""On the board (R83, the rowinv design's G1): every row of a prompt-pass product equals the same row in a one-row
call of it, across 32 and 128 rows; and tools/exl3_rows.py's copy of `_slab_rot_s1` equals the original.

Per pack (ROWINV_PACKS, comma list; the two Hy4 packs by default), for every distinct (N, Kin, bits) its trunk
multiplies by through `lin` in a prompt chunk (a real weight of each; Hy4 has no fused groups, so a group of two
trunk projections that share Kin and its member view are added), with x of 4,096 rows:

  the first M rows and the last M rows of x for M in 1, 2, 15, 16, 17, 31, 32, 33, 64, 74, 127, 128, 129, 130, 211,
  255, 256, 257, 349, 423, 511, 512, 513, 1,024, 4,095, 4,096: each sampled row (0, 1, 31, 32, 33, 127, 128, 129
  and the last of the call) equal to that row through a one-row call of the same arm

for each arm, each line naming the kernel class each M took:

  engine     the V2 chunk pass's product as the engine has it (`Hy4FlashEngine.<PROMPT_LIN>` once R83-hy4 adds it,
             `lin` before: today the row kernel to 32 rows, the skinny GEMM at `pick` to 128, the library above).
             Failing today, recorded; it is the test's verdict
  exl3_rows  tools/exl3_rows.linear: the skinny GEMM at S1 w8 to CROSS, the slab's s1 at L = KT / 8 above

and the copy: `exl3_rows.product` against `exl3_slab_gemm._slab_rot_s1` on the same rotated rows and slab, at 129 and
4,096 rows. One JSON line a (pack, shape, arm); the last line is the verdict.

Run on the box under the lock: python tests/gpu/test_linear_rowinv_gpu.py   (ROWINV_OUT=file: the lines there too)
"""

from __future__ import annotations

import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from tools import exl3_linear as X  # noqa: E402
from tools import exl3_rows as R  # noqa: E402

#: the name R83-hy4 gives the V2 chunk pass's product
PROMPT_LIN = "lin_prompt"
ROWS = (1, 2, 15, 16, 17, 31, 32, 33, 64, 74, 127, 128, 129, 130, 211, 255, 256, 257, 349, 423, 511, 512, 513, 1024,
        4095, 4096)
LOCAL = (0, 1, 31, 32, 33, 127, 128, 129)
N_X = 4096
HOME = os.path.expanduser("~")
PACKS = [p for p in os.environ.get("ROWINV_PACKS", f"{HOME}/models/hy4-flash-exl3-3.56,{HOME}/models/hy4-flash-exl3-3.08").split(",") if p]
SKIP = None if torch.cuda.is_available() and any(os.path.isdir(p) for p in PACKS) else "skipped: no device, or ROWINV_PACKS names no pack"


# ------------------------------------------------------------------------------------------ the pack's trunk
def trunk(pack_dir: str) -> tuple:
    """(pack, {(N, Kin, kb): [bases]}) for the EXL3 projections a Hy4 prompt chunk multiplies by through `lin`: the
    trunk's (the head, kv_b_proj (dequantised at load), the draft layers and the routed experts left out), in the
    pack's order."""
    from engine.models.hy4_flash.pack import Hy4Pack, projections
    pack = Hy4Pack.open(pack_dir)
    pack.audit()
    out: dict = {}
    for base, _, _ in projections(pack.cfg, experts=False, dfly_mlp=pack.dfly_mlp):
        if base == "lm_head" or base.endswith("kv_b_proj") or base.startswith("dfly_layers.") or ".experts." in base:
            continue
        if pack.formats.get(base) != "exl3":
            continue
        t = pack.tensors[base + ".trellis"]
        key = (t.shape[1] * 16, t.shape[0] * 16, X.kb_of(t.shape[2]))
        out.setdefault(key, []).append(base)
    return pack, out


def blocks(pack, shapes: dict, device="cuda") -> list:
    """[(label, EXL3Block)]: the first projection of each shape, then a group of the first two shapes that share Kin
    and the bits (as `EXL3Block.group` fuses one) and its second member as a view."""
    out = [(f"{N}x{K}x{kb} {bases[0]}", pack.exl3(bases[0], device)) for (N, K, kb), bases in shapes.items()]
    by_k: dict = {}
    for (N, K, kb), bases in shapes.items():
        by_k.setdefault((K, kb), []).append(bases[0])
    pair = next((v[:2] for v in by_k.values() if len(v) >= 2), None)
    if pair:
        g = X.EXL3Block.group([pack.exl3(b, device) for b in pair], list(pair))
        out += [(f"group {g.N}x{g.K}x{g.kb} {'+'.join(pair)}", g), (f"member 1 of the group {pair[1]}", g.member(1))]
    return out


def bare_engine():
    """A Hy4 engine object for its products alone (`lin`, `lin1`): the kernels on, no weights."""
    from engine.models.hy4_flash.engine import Hy4FlashEngine
    e = object.__new__(Hy4FlashEngine)
    e.kernels, e.block_rows, e.cuts = True, True, {}
    return e


def arms(eng) -> dict:
    prompt = getattr(eng, PROMPT_LIN, None)
    return {"engine": (prompt or eng.lin, PROMPT_LIN if prompt else "lin"), "exl3_rows": (R.linear, "exl3_rows.linear")}


def classes(arm: str, name: str, M: int) -> str:
    """The kernel class a call of M rows takes on the arm (as the code reads on this tree)."""
    if arm == "exl3_rows" or name == PROMPT_LIN:
        return "skinny S1 w8" if M <= R.CROSS else "slab s1"
    return "dense" if M <= R.ROWS else ("skinny pick" if M <= X.CHUNK_UNTIL else "slab library")


# ------------------------------------------------------------------------------------------ the comparison
def rows_check(fn, w, x: torch.Tensor, rows=ROWS, local=LOCAL) -> dict:
    """{M: (first-M rows equal, last-M rows equal)} against one-row calls of `fn` on each sampled row; and the first
    (M, row) that differs."""
    n = x.shape[0]
    want = sorted({r for M in rows for i in local if i < M for r in (i, n - M + i)} |
                  {M - 1 for M in rows} | {n - 1})
    one = {r: fn(x[r:r + 1].contiguous(), w) for r in want}
    got, first = {}, None
    for M in rows:
        res = []
        for base in (0, n - M):
            y = fn(x[base:base + M].contiguous(), w)
            ok = True
            for i in sorted({i for i in local if i < M} | {M - 1}):
                if not torch.equal(y[i:i + 1], one[base + i]):
                    ok = False
                    if first is None:
                        d = (y[i:i + 1].float() - one[base + i].float()).abs().max().item()
                        first = {"M": M, "rows": "first" if base == 0 else "last", "row": base + i, "max_abs": d}
            res.append(ok)
        got[M] = tuple(res)
    return {"rows": got, "first_differing": first}


def copy_check(w, x: torch.Tensor, rows=(129, N_X)) -> dict:
    """`exl3_rows.product` against `exl3_slab_gemm._slab_rot_s1` (its launch, at the same tile): bf16 and fp32."""
    from tools import exl3_slab_gemm as SG
    KT, L, Q = R.prompt_order(w).s1_launch()
    out = {}
    for M in rows:
        xh = X.rotate_in(x[:M].contiguous(), w)
        for dt in (torch.bfloat16, torch.float32):
            eq = True
            for g, n0, n1 in X.segments(w, X.SLAB_N):
                wi = w.inner(n0, n1)
                a = torch.empty(M, n1 - n0, dtype=dt, device=x.device)
                b = torch.empty_like(a)
                R.product(xh[g], wi, w.svh[n0:n1], a, KT, L, Q)
                bm, bn, nw, st = R.tile(w.K)
                grid = (triton_cdiv(M, bm) * ((n1 - n0) // bn),)
                SG._slab_rot_s1[grid](xh[g], wi, w.svh[n0:n1], b, M, n1 - n0, KT, L, xh[g].stride(0), wi.stride(0),
                                      b.stride(0), BM=bm, BN=bn, Q=Q, GROUP=8, OUT_F32=dt == torch.float32,
                                      num_warps=nw, num_stages=st)
                eq = eq and torch.equal(a, b)
            out[f"{M}.{str(dt).replace('torch.', '')}"] = eq
    return out


def triton_cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _say(line: dict, sink) -> None:
    s = json.dumps(line)
    print(s, flush=True)
    if sink is not None:
        sink.write(s + "\n")
        sink.flush()


def main() -> int:
    if SKIP:
        print(SKIP)
        return 0
    from engine.models.hy4_flash.load import apply_splits
    import triton
    applied = apply_splits()
    sink = open(os.environ["ROWINV_OUT"], "a") if os.environ.get("ROWINV_OUT") else None
    eng = bare_engine()
    A = arms(eng)
    verdict = {"engine": True, "exl3_rows": True, "copy": True}
    _say({"g1": "start", "packs": PACKS, "torch": torch.__version__, "triton": triton.__version__,
          "dense_splits_applied": applied, "cross": R.CROSS, "chunk_until": X.CHUNK_UNTIL, "rowgrid": X.ROWGRID,
          "engine_arm": A["engine"][1]}, sink)
    for pk in PACKS:
        if not os.path.isdir(pk):
            _say({"g1": "pack", "pack": pk, "skipped": "no such folder"}, sink)
            continue
        pack, shapes = trunk(pk)
        g = torch.Generator(device="cpu").manual_seed(5)
        for label, w in blocks(pack, shapes):
            try:
                _shape(pk, shapes, label, w, A, g, verdict, sink)
            except Exception as e:  # noqa: BLE001  (a shape that cannot run fails the verdict and is said)
                verdict["exl3_rows"] = False
                _say({"g1": "error", "pack": os.path.basename(pk), "shape": label, "error": f"{type(e).__name__}: {str(e)[:400]}"}, sink)
        torch.cuda.empty_cache()
    ok = all(verdict.values())
    _say({"g1": "verdict", "passed": ok, **verdict,
          "note": "the engine arm fails until R83-hy4; exl3_rows and copy must hold"}, sink)
    return 0 if ok else 1


def _shape(pk, shapes, label, w, A, g, verdict, sink) -> None:
    """One block: each arm's rows against its one-row calls, then the copy of s1 against the original."""
    x = (torch.randn(N_X, w.K, generator=g) * 0.5).to(torch.bfloat16).cuda()
    for arm, (fn, name) in A.items():
        try:
            r = rows_check(fn, w, x)
        except R.RowOrderError as e:            # the bind check refused the shape: unequal, and said why
            r = {"rows": {M: (False, False) for M in ROWS if M > R.ROWS}, "first_differing": {"bind": str(e)}}
        bad = sorted(M for M, v in r["rows"].items() if not all(v))
        verdict[arm] = verdict[arm] and not bad
        _say({"g1": "rows", "pack": os.path.basename(pk), "shape": label, "arm": arm, "fn": name,
              "equal": not bad, "unequal_M": bad, "first_differing": r["first_differing"],
              "classes": {str(M): classes(arm, name, M) for M in ROWS},
              "count": len(shapes.get((w.N, w.K, w.kb), []))}, sink)
    c = copy_check(w, x)
    verdict["copy"] = verdict["copy"] and all(c.values())
    _say({"g1": "copy", "pack": os.path.basename(pk), "shape": label, "equal": c}, sink)
    del x


def test_g1_rows_equal_their_one_row_calls():
    """For a runner that collects test functions: the whole check (needs the packs)."""
    if SKIP:
        return SKIP
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
