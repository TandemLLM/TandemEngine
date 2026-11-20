"""On the board (SPD-317, the prefill design's S0.3): the shared grouped expert kernel keeps the bytes it gives the
families that call it today.

`tools/fn_prefill_kernels.moe_grouped`, then `fn_kernels.moe_combine`, on seeded random banks at the expert shapes
of Flash-Next (2,560 to 640, routed K3, the shared expert at K5 as the last id), LFM2.5 (2,048 to 1,536, K3, no
shared expert in the bank) and Flint (2,048 to 512, K4, no shared expert in the bank); 64 routed experts a bank (the
count sets the item layout, not a pair's arithmetic); 1, 17, 300 and 4,096 rows; items of 32 and 64 rows. The
sha256 of every output (the fp32 pair outputs and the combined BF16 rows) is compared with the digests recorded on
the tree before the kernel's clamp argument (tests/byte_pins.py, fixture tests/fixtures/fn_moe_grouped_pin.json,
keyed by host, torch and device name). No call here passes the clamp: it is the call every family makes today.

    QSE_PIN_RECORD=1 python tests/gpu/test_fn_moe_grouped_pin_gpu.py     # once, on the tree before the change
    python tests/gpu/test_fn_moe_grouped_pin_gpu.py                       # every tree after
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from byte_pins import check, digest  # noqa: E402
from tools import exl3_linear as X  # noqa: E402
from tools import fn_kernels as FK  # noqa: E402
from tools import fn_prefill_kernels as PK  # noqa: E402

DEV = "cuda"
#: name -> (H, I, routed K, shared K or None, routed experts a row)
SHAPES = {"flashnext": (2560, 640, 3, 5, 10), "lfm25": (2048, 1536, 3, None, 4), "flint": (2048, 512, 4, None, 6)}
N_EXP = 64
ROWS = (1, 17, 300, 4096)
ITEMS = (32, 64)


def _bank(H: int, I: int, kb: int, shared_kb, seed: int):
    n = N_EXP + (1 if shared_kb is not None else 0)
    trip = []
    for e in range(n):
        k = shared_kb if e == N_EXP else kb
        blocks = (X.random_block(I, H, k, device=DEV, seed=seed + 3 * e), X.random_block(I, H, k, device=DEV, seed=seed + 3 * e + 1),
                  X.random_block(H, I, k, device=DEV, seed=seed + 3 * e + 2))
        trip.append(tuple((b.t, b.suh[0].contiguous(), b.svh) for b in blocks))
    return FK.MoEBank(trip, DEV)


def _inputs(M: int, H: int, k: int, shared: bool, seed: int):
    """x bf16 [M, H], ids int32 [M, k (+1)], weights fp32: drawn on the CPU, then moved (the same on every device)."""
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(M, H, generator=g) * 0.5).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(N_EXP, generator=g)[:k] for _ in range(M)])
    if shared:
        ids = torch.cat([ids, torch.full((M, 1), N_EXP, dtype=ids.dtype)], dim=1)
    w = torch.rand(M, ids.shape[1], generator=g).to(torch.bfloat16).float()
    return x.to(DEV), ids.to(torch.int32).to(DEV).contiguous(), w.to(DEV).contiguous()


def _digests(name: str) -> dict:
    H, I, kb, skb, k = SHAPES[name]
    bank = _bank(H, I, kb, skb, seed=1000 + 17 * kb)
    out = {}
    for M in ROWS:
        x, ids, w = _inputs(M, H, k, skb is not None, seed=M + 7 * kb)
        for items in ITEMS:
            yd = PK.moe_grouped(x, ids, bank, rows=items)
            out[f"rows{M}_items{items}_yd"] = digest(yd)
            out[f"rows{M}_items{items}_combined"] = digest(FK.moe_combine(yd, ids, w))
    torch.cuda.synchronize()
    return out


def _pin(name: str) -> str:
    got = _digests(name)
    assert got == _digests(name), f"{name}: two runs in one process differ, the pin cannot work"
    return check("fn_moe_grouped_pin.json", name, got, device=DEV)


def test_flash_next_shape():
    return _pin("flashnext")


def test_lfm25_shape():
    return _pin("lfm25")


def test_flint_shape():
    return _pin("flint")


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<32} ok   {fn() or ''}", flush=True)
        except AssertionError as e:
            fails += 1
            print(f"  {name:<32} FAIL {e}", flush=True)
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
