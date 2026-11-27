"""On the board: Hy4-flash row, prompt, and joint-pass kernels (SPD-345, SPD-347, SPD-349) against the operations they replace, bit for bit (no pack needed).

  * `hc` in one launch (`_hc_fused`) is the three launches' result at every row count, inject or not, head or not;
  * `mul_sigmoid` is x * torch.sigmoid(g) for every bf16 gate value;
  * `rot_exact` is ref.py's `rotate` (the prompt pass's rotary) on query, latent and indexer shapes;
  * `engram_decode` is table.py's `decode_rows(...).to(bf16)` (the host decode) for K 2 to 5;
  * `gather_lanes` and `mm_lanes` (the joint pass's reads by lane address) are the per-lane gather and `mm`.

Run on the box: python tests/gpu/test_hy4_joint_kernels_gpu.py   (under flock ~/.qwen38-box.flock and ops/hold.sh)
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

SKIP = None if torch.cuda.is_available() else "skipped: no device"
DEV = "cuda"


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def test_hc_in_one_launch_is_the_three_launches():
    if SKIP:
        return SKIP
    from engine.models.hy4_flash import kernels as K_
    g = torch.Generator(device=DEV).manual_seed(0)
    H, S, n = 3072, 4, 0
    for B in (1, 3, 16, 33, 1024):
        for head in (False, True):
            for inj in (False, True):
                nmix = S if head else 2 * S
                h = (torch.randn(B, S, H, device=DEV, generator=g) * 2).to(torch.bfloat16)
                y = torch.randn(B, H, device=DEV, generator=g).to(torch.bfloat16)
                post = torch.rand(B, S, device=DEV, generator=g) * 2
                fn = torch.randn(nmix, S * H, device=DEV, generator=g) * 0.02
                base, scale = torch.randn(nmix, device=DEV, generator=g) * 0.1, torch.rand(3, device=DEV, generator=g)
                w = (torch.rand(H, device=DEV, generator=g) + 0.5).to(torch.bfloat16)
                pend = (y, post) if inj else None
                old = K_.HC_FUSED_MIN
                try:
                    K_.HC_FUSED_MIN = 0
                    a = K_.hc(h.clone(), pend, fn, base, scale, w, 1e-6, 1e-6, 2.0, head=head)
                    K_.HC_FUSED_MIN = 1
                    b = K_.hc(h.clone(), pend, fn, base, scale, w, 1e-6, 1e-6, 2.0, head=head)
                finally:
                    K_.HC_FUSED_MIN = old
                assert _bits_equal(a[0], b[0]) and _bits_equal(a[2], b[2]), (B, head, inj)
                assert head or _bits_equal(a[1], b[1]), (B, head, inj)
                n += 1
    return f"{n} shapes, every output equal"


def test_mul_sigmoid_is_torchs_for_every_gate():
    if SKIP:
        return SKIP
    from engine.models.hy4_flash import kernels as K_
    if not K_.mul_sigmoid_ok():
        return "skipped: no libdevice exp / div_rn in this Triton"
    gate = torch.arange(-32768, 32768, dtype=torch.int32, device=DEV).to(torch.int16).view(torch.bfloat16).repeat(4)
    x = (torch.randn(gate.numel(), device=DEV, generator=torch.Generator(device=DEV).manual_seed(1)) * 3).to(torch.bfloat16)
    want, got = x * torch.sigmoid(gate), K_.mul_sigmoid(x, gate)
    both_nan = torch.isnan(want) & torch.isnan(got)
    bad = int(((want.view(torch.int16) != got.view(torch.int16)) & ~both_nan).sum())
    assert bad == 0, bad
    return f"{gate.numel()} values equal"


def test_rot_exact_is_rotate():
    if SKIP:
        return SKIP
    from engine.models.hy4_flash import kernels as K_
    from engine.models.hy4_flash.ref import rotate
    g = torch.Generator(device=DEV).manual_seed(2)
    for T, H, D, d0 in ((1, 64, 256, 192), (4096, 64, 256, 192), (4096, 1, 576, 512), (513, 4, 128, 64)):
        x = (torch.randn(T, H, D, device=DEV, generator=g) * 3).to(torch.bfloat16)
        P = (D - d0) // 2
        f = torch.randint(0, 200000, (T,), device=DEV, generator=g).float()[:, None] * (10000000.0 ** (-torch.arange(P, device=DEV).float() / P))
        cos, sin = f.cos().to(torch.bfloat16).contiguous(), f.sin().to(torch.bfloat16).contiguous()
        want = torch.cat((x[..., :d0], rotate(x[..., d0:], cos[:, None], sin[:, None])), dim=-1)
        got = x.clone()
        K_.rot_exact(got, cos, sin, d0)
        assert _bits_equal(want, got), (T, H, D)
    return "4 shapes equal"


def test_engram_decode_is_the_host_decode():
    if SKIP:
        return SKIP
    from engine.models.hy4_flash import kernels as K_
    from engine.models.hy4_flash.table import decode_rows, mul1_codebook
    cb = mul1_codebook("cpu")
    gen = torch.Generator().manual_seed(3)
    for K in (2, 3, 4, 5):
        dim, N = 96, 4096
        W = 1 + dim * K // 16
        pk = torch.randint(-32768, 32768, (N, W), dtype=torch.int32, generator=gen).to(torch.int16)
        pk[:, 0] = (torch.rand(N, generator=gen) * 0.05 + 0.001).to(torch.float16).view(torch.int16)
        heads = torch.randint(0, 8, (N,), generator=gen)
        bias = (torch.randn(8, dim, generator=gen) * 0.1).to(torch.float16)
        want = decode_rows(pk, heads, bias, K, cb, dim).to(torch.bfloat16)
        got = K_.engram_decode(pk.to(DEV), heads.to(torch.int32).to(DEV), bias.to(DEV), K, cb.to(DEV), dim).cpu()
        assert _bits_equal(want, got), K
    return "K 2..5 equal"


def test_lane_reads_are_the_per_lane_reads():
    if SKIP:
        return SKIP
    from engine.models.hy4_flash import kernels as K_
    g = torch.Generator(device=DEV).manual_seed(4)
    lanes = [torch.randn(3, 700, 576, device=DEV, generator=g).to(torch.bfloat16) for _ in range(3)]
    scratch = torch.randn(40, 576, device=DEV, generator=g).to(torch.bfloat16)
    R, n = 13, 300
    lane = torch.randint(0, 3, (R,), device=DEV, generator=g)
    idx = torch.randint(0, 700, (R, n), device=DEV, generator=g)
    flag = torch.rand(R, n, device=DEV, generator=g) < 0.2
    idx = torch.where(flag, idx % 40, idx).contiguous()
    ptr = torch.tensor([t.data_ptr() for t in lanes], dtype=torch.int64, device=DEV)
    got = K_.gather_lanes(idx, flag.contiguous(), lane, ptr, 1 * 700 * 576, 576, scratch, 576)
    want = torch.stack([torch.where(flag[r, :, None], scratch[idx[r].clamp(0, 39)], lanes[int(lane[r])][1][idx[r].clamp(0, 699)])
                        for r in range(R)])
    assert _bits_equal(want, got)
    keys = [torch.randn(2, 600, 128, device=DEV, generator=g).to(torch.bfloat16) for _ in range(3)]
    kptr = torch.tensor([t.data_ptr() for t in keys], dtype=torch.int64, device=DEV)
    a = torch.randn(R, 4, 128, device=DEV, generator=g).to(torch.bfloat16)
    got = K_.mm_lanes(a, kptr, lane, 1 * 600 * 128, 128, 517, 1, 128, 64)
    want = torch.cat([K_.mm(a[r:r + 1], keys[int(lane[r])][1][:517].t().unsqueeze(0), 64) for r in range(R)])
    assert _bits_equal(want, got)
    return "gather and mm by lane address equal the per-lane ones"


def test_attend_r_wk_transpose_is_bit_equal():
    if SKIP:
        return SKIP
    from engine.models.hy4_flash import kernels as K_
    g = torch.Generator(device=DEV).manual_seed(5)
    H, nope, R, V = 64, 192, 512, 256
    wk = (torch.randn(H, nope, R, device=DEV, generator=g) * 0.1).to(torch.bfloat16)
    wv = (torch.randn(H, R, V, device=DEV, generator=g) * 0.1).to(torch.bfloat16)
    for M in (1, 5, 12, 24, 32, 40):
        q = (torch.randn(M, H, nope, device=DEV, generator=g) * 0.5).to(torch.bfloat16)
        want = K_.mm(q.reshape(M * H, 1, nope), wk).view(M, H, R).to(torch.bfloat16)
        got = K_.mm(q.transpose(0, 1), wk).to(torch.bfloat16).transpose(0, 1)
        assert _bits_equal(want, got), ("wk", M)
        mix = (torch.randn(M, H, R, device=DEV, generator=g) * 0.5).to(torch.bfloat16)
        want_v = K_.mm(mix.reshape(M * H, 1, R), wv).view(M, H, V).to(torch.bfloat16)
        got_v = K_.mm(mix.transpose(0, 1), wv).transpose(0, 1).to(torch.bfloat16)
        assert _bits_equal(want_v, got_v), ("wv", M)
    return "M=1,5,12,24,32,40 equal"


def test_select_gcap_and_qi_reshape_is_bit_equal():
    if SKIP:
        return SKIP
    from engine.models.flashnext.ref import topk_stable
    from engine.models.hy4_flash import kernels as K_
    from engine.models.hy4_flash.engine import GCAP_BUCKETS
    from engine.models.hy4_flash.ref import rotate
    g = torch.Generator(device=DEV).manual_seed(6)
    Gc, BG, nh, D, rd, gt, rr = 65536, 9, 4, 128, 64, 512, 4
    gk = (torch.randn(Gc + BG, D, device=DEV, generator=g) * 0.2).to(torch.bfloat16)
    bgk = (torch.randn(32, D, device=DEV, generator=g) * 0.2).to(torch.bfloat16)
    gk_ptr = torch.tensor([gk.data_ptr()], dtype=torch.int64, device=DEV)
    for ctx in (2200, 3538, 8500):
        M = 6
        iq = (torch.randn(M, nh * D, device=DEV, generator=g) * 0.2).to(torch.bfloat16)
        hw = (torch.randn(M, nh, device=DEV, generator=g) * 0.2).to(torch.bfloat16)
        pos = torch.arange(ctx, ctx + M, device=DEV)
        qg = pos // rr
        gmax = torch.clamp(qg, max=ctx // rr)
        f = pos.float()[:, None] * (10000000.0 ** (-torch.arange(rd // 2, device=DEV).float() / (rd // 2)))
        cos, sin = f.cos().to(torch.bfloat16).contiguous(), f.sin().to(torch.bfloat16).contiguous()
        cidx = torch.zeros(M, BG, dtype=torch.int64, device=DEV)
        bgid = torch.full((M, BG), ctx // rr, dtype=torch.int64, device=DEV)
        bok = torch.zeros(M, BG, dtype=torch.bool, device=DEV)
        bok[3:, 0] = True
        wsc = (hw.float() * (nh ** -0.5 * D ** -0.5)).contiguous()
        # reference: full Gc=65536 + Python rotate + per-row mm
        qi0 = iq.view(M, nh, D)
        qi0 = torch.cat((qi0[..., :D - rd], rotate(qi0[..., D - rd:], cos[:, None], sin[:, None])), dim=-1)
        sc0 = torch.cat((K_.mm(qi0, gk[:Gc].t().unsqueeze(0), 64), K_.mm(qi0, bgk[cidx].transpose(1, 2), 64)), dim=-1)
        lg0 = K_.wsum_relu(sc0, wsc)
        gid0 = torch.arange(Gc + BG, device=DEV)
        ok0 = (gid0[None, :] < gmax[:, None]) | ((gid0[None, :] >= Gc) & bok.gather(1, (gid0 - Gc).clamp(0, BG - 1).expand(M, -1)))
        top0 = topk_stable(lg0.masked_fill(~ok0, float("-inf")), gt - 1)[1]
        want = torch.where(top0 < Gc, top0, bgid.gather(1, (top0 - Gc).clamp(0, BG - 1))).sort(dim=-1).values
        # candidate single-lane: gcap + rot_exact + [1, M*nh, D] mm
        gcap = min(Gc, next(b for b in GCAP_BUCKETS if b >= max(gt, int(gmax.max()))))
        qi1 = iq.reshape(M, nh, D).contiguous()
        K_.rot_exact(qi1, cos, sin, D - rd)
        sc1 = torch.cat((K_.mm(qi1.reshape(1, M * nh, D), gk[:gcap].t().unsqueeze(0), 64).view(M, nh, gcap),
                         K_.mm(qi1, bgk[cidx].transpose(1, 2), 64)), dim=-1)
        lg1 = K_.wsum_relu(sc1, wsc)
        gid1 = torch.arange(gcap + BG, device=DEV)
        ok1 = (gid1[None, :] < gmax[:, None]) | ((gid1[None, :] >= gcap) & bok.gather(1, (gid1 - gcap).clamp(0, BG - 1).expand(M, -1)))
        top1 = topk_stable(lg1.masked_fill(~ok1, float("-inf")), gt - 1)[1]
        got1 = torch.where(top1 < gcap, top1, bgid.gather(1, (top1 - gcap).clamp(0, BG - 1))).sort(dim=-1).values
        assert _bits_equal(want, got1), ("single", ctx, gcap)
        # candidate multi-lane: gcap + mm_lanes
        lane = torch.zeros(M, dtype=torch.int64, device=DEV)
        sc2 = torch.cat((K_.mm_lanes(qi1, gk_ptr, lane, 0, D, gcap, 1, D, 64),
                         K_.mm(qi1, bgk[cidx].transpose(1, 2), 64)), dim=-1)
        lg2 = K_.wsum_relu(sc2, wsc)
        top2 = topk_stable(lg2.masked_fill(~ok1, float("-inf")), gt - 1)[1]
        got2 = torch.where(top2 < gcap, top2, bgid.gather(1, (top2 - gcap).clamp(0, BG - 1))).sort(dim=-1).values
        assert _bits_equal(want, got2), ("multi", ctx, gcap)
    return "ctx=2200,3538,8500 gcap vs Gc=65536 equal"


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                print(f"  {name:<52} ok   {fn() or ''}", flush=True)
                passed += 1
            except AssertionError as e:
                print(f"  {name:<52} FAIL {e}", flush=True)
                failed += 1
    print(f"{passed} passed" + (f", {failed} FAILED" if failed else ""))
    sys.exit(1 if failed else 0)
