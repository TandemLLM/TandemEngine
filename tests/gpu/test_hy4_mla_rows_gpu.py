"""On the board (SPD-323, the prefill design's F1): Hy4-flash's prompt-pass attention kernel `mla_rows`.

  * against its torch reference within MLA_REL (the kernel's weights are rounded to bf16 before the mix), at the
    packs' shapes (latents of 512 + 64, 32 heads of the sparse layers and 64 of the window layers), for rows that read
    every row up to theirs, a window of 512 rows, and a list of 2,048 rows;
  * a row's bits are its own: each of a few rows alone equals the same row among 300;
  * its bytes on fixed inputs equal the digests recorded when the prompt pass flips to it (tests/byte_pins.py,
    fixture tests/fixtures/hy4_mla_rows_pin.json): a later change of a launch setting must keep them.

    QSE_PIN_RECORD=1 python tests/gpu/test_hy4_mla_rows_gpu.py      # once, at the flip
    python tests/gpu/test_hy4_mla_rows_gpu.py
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from byte_pins import check, digest  # noqa: E402
from engine.models.hy4_flash import kernels as K  # noqa: E402

DEV = "cuda"
R, RP, N, T = 512, 64, 9000, 300
MLA_REL = 1e-2      # this builder's bound (two bf16 steps of the weights), not the design's


def _inputs(heads: int, seed: int) -> tuple:
    """Drawn on the CPU (the same on every device), then moved: latents of N positions, T rows at positions in
    [N - 2,000, N): a third dense from 0, a third a window of 512, a third a list of 2,048 ascending (their own last)."""
    g = torch.Generator().manual_seed(seed)
    lat = (torch.randn(N, R + RP, generator=g) * 0.5).to(torch.bfloat16)
    qa = (torch.randn(T, heads, R, generator=g) * 0.05).to(torch.bfloat16)
    qr = (torch.randn(T, heads, RP, generator=g) * 0.05).to(torch.bfloat16)
    pos = torch.randint(N - 2000, N, (T,), generator=g)
    ctl = torch.zeros(T, 3, dtype=torch.int32)
    sel = torch.zeros(T, 2048, dtype=torch.int32)
    for t in range(T):
        p = int(pos[t])
        if t % 3 == 0:
            ctl[t] = torch.tensor([0, p + 1, 0])
        elif t % 3 == 1:
            ctl[t] = torch.tensor([p - 511, 512, 0])
        else:
            keep = torch.randperm(p, generator=g)[:2047].sort().values.tolist() + [p]
            ctl[t] = torch.tensor([0, 2048, 1])
            sel[t] = torch.tensor(keep, dtype=torch.int32)
    sink = torch.randn(heads, generator=g)
    return [x.to(DEV) for x in (qa, qr, lat, ctl, sel, sink)]


def _rel(a, b):
    return float((a.float() - b.float()).abs().max() / b.float().abs().max())


def test_mla_rows_against_the_reference():
    out = []
    for heads, qk in ((32, 256), (64, 256)):
        qa, qr, lat, ctl, sel, sink = _inputs(heads, heads)
        got = K.mla_rows(qa, qr, lat, ctl, sel, sink, qk ** -0.5)
        want = K.mla_rows_reference(qa, qr, lat, ctl, sel, sink, qk ** -0.5)
        r = _rel(got, want)
        assert r < MLA_REL, (heads, r)
        out.append(f"{heads} heads {r:.1e}")
    return "dense, window and list rows: " + ", ".join(out)


def test_a_rows_bits_are_its_own():
    qa, qr, lat, ctl, sel, sink = _inputs(64, 3)
    full = K.mla_rows(qa, qr, lat, ctl, sel, sink, 256 ** -0.5)
    for j in (0, 1, 2, 150, 299):
        alone = K.mla_rows(qa[j:j + 1], qr[j:j + 1], lat, ctl[j:j + 1], sel[j:j + 1], sink, 256 ** -0.5)
        assert torch.equal(alone[0], full[j]), j
    return "5 rows alone equal the same rows among 300"


def test_the_block_select_takes_the_own_group():
    """F3: the shared block select's kernels (a bitonic sort up to 8,192 blocks, a radix select past it) on scores with
    a row's own group at +inf and the groups past it at -inf (Flash-Next never feeds them +inf) equal its torch form."""
    from tools import fn_seq_kernels as SK
    out = []
    for G in (600, 9000):
        g = torch.Generator().manual_seed(G)
        t = 64
        z = torch.randint(0, 40, (t, G + 1), generator=g).float() / 7
        qg = torch.randint(511, G + 1, (t,), generator=g)
        col = torch.arange(G + 1)
        z = z.masked_fill(col[None, :] > qg[:, None], float("-inf")).masked_fill(col[None, :] == qg[:, None], float("inf"))
        z, tail = z.to(DEV).contiguous(), torch.zeros(t, 3, dtype=torch.int32, device=DEV)
        a = SK.qsa_select(z, tail, torch.empty(t, 4 * SK.TOPK + 3, dtype=torch.int32, device=DEV))
        b = SK.qsa_select_torch(z, tail, torch.empty(t, 4 * SK.TOPK + 3, dtype=torch.int32, device=DEV))
        assert torch.equal(a[:, :4 * SK.TOPK], b[:, :4 * SK.TOPK]), G
        out.append(f"{G} groups")
    return ", ".join(out) + ": kernel equals torch with +inf own groups"


def test_the_flip_digests():
    got = {}
    for heads in (32, 64):
        qa, qr, lat, ctl, sel, sink = _inputs(heads, 100 + heads)
        got[f"heads{heads}"] = digest(K.mla_rows(qa, qr, lat, ctl, sel, sink, 256 ** -0.5))
    launch = f"HB{K.MLA_HB}-BN{K.MLA_BN}-W{K.MLA_WARPS}-S{K.MLA_STAGES}"
    return check("hy4_mla_rows_pin.json", "mla_rows", got, device=DEV) + f" ({launch})"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<40} ok   {fn() or ''}", flush=True)
        except AssertionError as e:
            fails += 1
            print(f"  {name:<40} FAIL {e}", flush=True)
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
